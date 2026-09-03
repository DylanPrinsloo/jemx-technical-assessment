import polars as pl
from datetime import datetime

# 1. Reverse-engineer clock duration math for Week 1 (2026-06-08)
shifts = pl.read_csv("data/shifts.csv")
summary = pl.read_csv("data/weekly_summary.csv")

def parse_duration(clock_in, clock_out):
    if clock_in is None or clock_out is None:
        return 0.0
    t_in = datetime.strptime(clock_in, "%H:%M")
    t_out = datetime.strptime(clock_out, "%H:%M")
    diff = (t_out - t_in).total_seconds() / 3600.0
    if diff < 0:  # Crosses midnight
        diff += 24.0
    return diff

# Calculate raw hours for E1001 and E1002 in Week 1
w1_shifts = shifts.filter(
    (pl.col("shift_date") >= "2026-06-08") & 
    (pl.col("shift_date") <= "2026-06-14")
)

print("--- CLOCK TIME MATH VERIFICATION ---")
for emp_id in ["E1001", "E1002", "E1004", "E1005"]:
    emp_s = w1_shifts.filter(pl.col("employee_id") == emp_id)
    raw_hrs = sum(parse_duration(r["clock_in_time"], r["clock_out_time"]) for r in emp_s.iter_rows(named=True))
    sys_row = summary.filter((pl.col("employee_id") == emp_id) & (pl.col("week_starting") == "2026-06-08"))
    sys_hrs = sys_row["total_hours"][0] if len(sys_row) > 0 else None
    print(f"{emp_id}: Raw sum = {raw_hrs:.2f}h | weekly_summary total_hours = {sys_hrs}h | Diff = {raw_hrs - (sys_hrs or 0):.2f}h")

# 2. Inspect typical Thursday-Sunday shift patterns
print("\n--- SHIFT DISTRIBUTION BY DAY OF WEEK ---")
shifts_dow = shifts.filter(pl.col("shift_date") < "2026-08-10").with_columns(
    pl.col("shift_date").str.to_date("%Y-%m-%d").dt.weekday().alias("dow")  # 1=Mon, 7=Sun
)
dow_counts = shifts_dow.group_by("dow").len().sort("dow")
dow_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
for r in dow_counts.iter_rows(named=True):
    print(f"{dow_names[r['dow']]}: {r['len']} shifts")

# 3. Current week standing as of Wednesday night
cur_summary = summary.filter(pl.col("week_starting") == "2026-08-10")
print(f"\nCurrent week standing (up to Wednesday):")
print(f"Top 5 accumulated hours by Wednesday:")
print(cur_summary.sort("total_hours", descending=True).head(5))