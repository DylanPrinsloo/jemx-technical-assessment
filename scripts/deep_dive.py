import polars as pl

shifts = pl.read_csv("data/shifts.csv").with_columns(
    pl.col("shift_date").str.to_date("%Y-%m-%d")
)
notes = pl.read_csv("data/shift_notes.csv")
emp = pl.read_csv("data/employees.csv")

# 1. Investigate the 3 missing clock-outs in current week
print("--- MISSING CLOCK-OUTS IN CURRENT WEEK ---")
cur_shifts = shifts.filter(pl.col("shift_date") >= pl.date(2026, 8, 10))
missing = cur_shifts.filter(pl.col("clock_out_time").is_null())
print(missing.join(emp, on="employee_id", how="left").select(
    ["shift_id", "employee_id", "full_name", "role", "shift_date", "clock_in_time", "shift_pattern"]
))

# 2. Check typical shift duration by role across the entire dataset
print("\n--- MEDIAN SHIFT LENGTH BY ROLE ---")
def get_hours(df):
    return df.filter(pl.col("clock_out_time").is_not_null()).with_columns(
        (
            pl.col("clock_out_time").str.to_time("%H:%M") - 
            pl.col("clock_in_time").str.to_time("%H:%M")
        ).dt.total_minutes().alias("mins")
    ).with_columns(
        pl.when(pl.col("mins") < 0)
        .then(pl.col("mins") + 24 * 60)
        .otherwise(pl.col("mins")) / 60.0
    )

valid_shifts = get_hours(shifts)
median_by_role = valid_shifts.join(emp, on="employee_id", how="left").group_by("role").agg(
    pl.col("mins").median().alias("median_shift_hours"),
    pl.col("mins").mean().alias("mean_shift_hours")
)
print(median_by_role)

# 3. Sample 20 random notes to understand language & reasons
print("\n--- SAMPLE SHIFT NOTES ---")
print(notes.sample(15, seed=42)["note"].to_list())