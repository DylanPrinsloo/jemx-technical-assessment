import polars as pl
import pandas as pd
from datetime import datetime

print("--- SHIFTS INSPECTION ---")
shifts = pl.read_csv("data/shifts.csv")
print(f"Total shift records: {len(shifts)}")
print(f"Missing clock-out times: {shifts['clock_out_time'].null_count()}")

# Parse shift_date to find latest date and day of week
shifts = shifts.with_columns(pl.col("shift_date").str.to_date("%Y-%m-%d"))
max_date = shifts["shift_date"].max()
min_date = shifts["shift_date"].min()
print(f"Date range: {min_date} to {max_date} ({max_date.strftime('%A')})")

print("\n--- EMPLOYEE REGISTER ANOMALIES ---")
emp = pl.read_csv("data/employees.csv")
print(f"Total registered employees: {len(emp)}")
dup_ids = emp.filter(pl.col("id_number").is_duplicated()).sort("id_number")
print(f"Duplicate SA ID numbers found: {len(dup_ids)}")
if len(dup_ids) > 0:
    print(dup_ids.select(["employee_id", "full_name", "id_number", "primary_site_id", "employment_type"]))

print("\n--- SHIFT NOTES INSPECTION ---")
notes = pl.read_csv("data/shift_notes.csv")
print(f"Total shift notes: {len(notes)}")
print(notes.head(5))

print("\n--- FILES IN DATA/ ---")
for name in ["sites.csv", "public_holidays.csv", "weekly_summary.csv", "payroll_details.csv"]:
    df = pl.read_csv(f"data/{name}")
    print(f"{name}: {df.shape[0]} rows, cols: {df.columns}")