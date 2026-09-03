import polars as pl

# 1. Reverse engineer weekly_summary logic
summary = pl.read_csv("data/weekly_summary.csv")
print("--- WEEKLY SUMMARY SAMPLES ---")
print(summary.head(5))

# Check breach rule
breached = summary.filter(pl.col("breached") == 1)
not_breached = summary.filter(pl.col("breached") == 0)
print(f"\nTotal historical summary rows: {len(summary)}")
print(f"Breached rows: {len(breached)}")
print(f"Min overtime in breached: {breached['overtime_hours'].min()}")
print(f"Max overtime in not_breached: {not_breached['overtime_hours'].max()}")
print(f"Min total_hours in breached: {breached['total_hours'].min()}")
print(f"Max total_hours in not_breached: {not_breached['total_hours'].max()}")

# 2. Check if current week is in weekly_summary
print(f"\nDistinct week_starting in weekly_summary: {summary['week_starting'].unique().sort().to_list()}")

# 3. Check missing clock outs in the current week (2026-08-10 to 2026-08-12)
shifts = pl.read_csv("data/shifts.csv").with_columns(pl.col("shift_date").str.to_date("%Y-%m-%d"))
cur_shifts = shifts.filter(pl.col("shift_date") >= pl.date(2026, 8, 10))
print(f"\nCurrent week shift count: {len(cur_shifts)}")
print(f"Current week missing clock-outs: {cur_shifts['clock_out_time'].null_count()}")

# 4. Check duplicate employees in weekly_summary
dup_ids = ["E1035", "E1036"]
print(f"\nHistory for duplicate pair E1035 / E1036:")
print(summary.filter(pl.col("employee_id").is_in(dup_ids)).sort(["week_starting", "employee_id"]))