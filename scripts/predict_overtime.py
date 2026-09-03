import polars as pl
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# 1. Load Data
shifts = pl.read_csv("data/shifts.csv")
employees = pl.read_csv("data/employees.csv")
weekly_summary = pl.read_csv("data/weekly_summary.csv")
notes = pl.read_csv("note_classifications.csv")

# 2. Impute Durations (Role-conditioned: 12h for Guards, 9h for Day Workers)
shifts = shifts.join(employees.select(["employee_id", "role", "shift_pattern"]), on="employee_id")

# Parse timestamps & calculate duration with overnight rollover
shifts = shifts.with_columns(
    pl.concat_str([pl.col("shift_date"), pl.col("clock_in_time")], separator=" ").str.to_datetime("%Y-%m-%d %H:%M").alias("in_dt"),
    pl.concat_str([pl.col("shift_date"), pl.col("clock_out_time")], separator=" ").str.to_datetime("%Y-%m-%d %H:%M").alias("out_dt"),
).with_columns(
    pl.when(pl.col("out_dt") < pl.col("in_dt")).then(pl.col("out_dt") + pl.duration(days=1)).otherwise(pl.col("out_dt")).alias("out_dt")
).with_columns(
    ((pl.col("out_dt") - pl.col("in_dt")).dt.total_minutes() / 60.0).alias("duration_hrs")
).with_columns(
    pl.when(pl.col("duration_hrs").is_not_null())
    .then(pl.col("duration_hrs"))
    .when(pl.col("role") == "Security Guard")
    .then(pl.lit(12.0))
    .otherwise(pl.lit(9.0))
    .alias("duration_hrs")
)

# Attach calendar metadata
shifts = shifts.with_columns(
    pl.col("in_dt").dt.truncate("1w").dt.to_string("%Y-%m-%d").alias("week_starting"),
    pl.col("in_dt").dt.weekday().alias("weekday") # 1=Mon, 2=Tue, 3=Wed, ..., 7=Sun
)

# 3. Simulate Historical Wednesday Cutoffs (Weeks 1 to 9)
hist_shifts = shifts.filter(pl.col("shift_date") < "2026-08-10")

# Actual Mon-Wed hours per employee-week
wed_cutoff = hist_shifts.filter(pl.col("weekday") <= 3).group_by(["week_starting", "employee_id"]).agg(
    pl.col("duration_hrs").sum().alias("hours_mon_wed")
)

# Join actual ground truth from weekly_summary
backtest_df = weekly_summary.join(wed_cutoff, on=["week_starting", "employee_id"], how="left").with_columns(
    pl.col("hours_mon_wed").fill_null(0.0)
).join(employees.select(["employee_id", "role", "shift_pattern"]), on="employee_id")

# Baselines for Video Comparison
# Baseline 1: Linear Extrapolation (Hours * 7/3 > 55.0)
backtest_df = backtest_df.with_columns(
    (pl.col("hours_mon_wed") * (7.0 / 3.0) > 55.0).cast(pl.Int64).alias("pred_linear")
)

# Calculate Role-Based Expected Weekend Hours from History
role_weekend_pace = hist_shifts.filter(pl.col("weekday") > 3).group_by(["role", "shift_pattern"]).agg(
    pl.col("duration_hrs").sum().alias("total_weekend_hrs"),
    pl.col("employee_id").n_unique().alias("num_emps")
).with_columns(
    (pl.col("total_weekend_hrs") / (pl.col("num_emps") * 9.0)).alias("avg_thu_sun_hrs")
)

backtest_df = backtest_df.join(role_weekend_pace.select(["role", "shift_pattern", "avg_thu_sun_hrs"]), on=["role", "shift_pattern"])
backtest_df = backtest_df.with_columns(
    (pl.col("hours_mon_wed") + pl.col("avg_thu_sun_hrs")).alias("projected_total")
).with_columns(
    (1.0 / (1.0 + np.exp(-(pl.col("projected_total") - 55.0) / 3.5))).alias("risk_score")
).with_columns(
    (pl.col("risk_score") >= 0.50).cast(pl.Int64).alias("pred_model")
)

print("--- BASELINE 1: LINEAR PACE EVALUATION ---")
print(classification_report(backtest_df["breached"].to_numpy(), backtest_df["pred_linear"].to_numpy(), zero_division=0))

print("\n--- MODEL: ROLE-PACED SIGMOID EVALUATION ---")
print(classification_report(backtest_df["breached"].to_numpy(), backtest_df["pred_model"].to_numpy(), zero_division=0))
print("ROC-AUC:", roc_auc_score(backtest_df["breached"].to_numpy(), backtest_df["risk_score"].to_numpy()))
print("PR-AUC:", average_precision_score(backtest_df["breached"].to_numpy(), backtest_df["risk_score"].to_numpy()))

# 4. Predict Active Target Week (2026-08-10 to 2026-08-16)
target_shifts = shifts.filter(pl.col("shift_date") >= "2026-08-10")
target_hours = target_shifts.group_by("employee_id").agg(
    pl.col("duration_hrs").sum().alias("hours_mon_wed")
)

# Roster all 213 employees from employees.csv (including those with 0 hours Mon-Wed)
preds = employees.select(["employee_id", "role", "shift_pattern"]).join(
    target_hours, on="employee_id", how="left"
).with_columns(
    pl.col("hours_mon_wed").fill_null(0.0)
).join(
    role_weekend_pace.select(["role", "shift_pattern", "avg_thu_sun_hrs"]), on=["role", "shift_pattern"]
).with_columns(
    (pl.col("hours_mon_wed") + pl.col("avg_thu_sun_hrs")).alias("projected_total")
).with_columns(
    (1.0 / (1.0 + np.exp(-(pl.col("projected_total") - 55.0) / 3.5))).round(2).alias("risk_score")
).with_columns(
    (pl.col("risk_score") >= 0.50).cast(pl.Int64).alias("will_breach")
)

# Export predictions.csv matching exact specification
output_preds = preds.select(["employee_id", "will_breach", "risk_score"])
output_preds.write_csv("predictions.csv")
print(f"\nExported predictions.csv: {len(output_preds)} rows")
print("Projected Breaches for Target Week:", output_preds["will_breach"].sum())