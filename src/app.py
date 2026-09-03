import datetime
import re
import numpy as np
import polars as pl
import streamlit as st

st.set_page_config(page_title="Ops Control: Overtime Triage", layout="wide")

# -------------------------------------------------------------
# 1. Core Classification & Parsing Engine
# -------------------------------------------------------------
PATTERNS = {
    "relief_no_show": re.compile(
        r"(not?\s+pitch|didn'?t\s+come|no[- ]show|stood\s+in|covered|replacement|relie[fv]|akafikanga|akezanga|ngimele|afwesig|geen\s+aflos|double\s+shift)",
        re.IGNORECASE,
    ),
    "equipment_failure": re.compile(
        r"(generator|masjien|stukkend|broken|fault|power\s+outage|load[\s-]?shedding|gate\s+fault|alarm|water\s+leak|herstel|krag)",
        re.IGNORECASE,
    ),
    "handover_delay": re.compile(
        r"(handover|oorhandiging|keys?|sleutel|lock\s+box|missing\s+keys)",
        re.IGNORECASE,
    ),
    "client_request": re.compile(
        r"(client\s+(asked|wanted|requested|signed|email)|approved|customer\s+request|klient|gemagtig|extra\s+(patrol|man|guard|hours|shift)|stocktake)",
        re.IGNORECASE,
    ),
}

ROUTINE_SUBSTRINGS = re.compile(
    r"(all\s+good|quiet\s+shift|sharp|ok|normal\s+shift|alles\s+reg|lungile|routine|nothing\s+to\s+report)",
    re.IGNORECASE,
)


def classify_text(note: str | None) -> str:
    if not note or not str(note).strip():
        return "routine_or_unclear"
    text = str(note).strip()
    if ROUTINE_SUBSTRINGS.search(text) and not any(
        p.search(text) for p in PATTERNS.values()
    ):
        return "routine_or_unclear"
    if PATTERNS["relief_no_show"].search(text):
        return "relief_no_show"
    if PATTERNS["equipment_failure"].search(text):
        return "equipment_failure"
    if PATTERNS["handover_delay"].search(text):
        return "handover_delay"
    if PATTERNS["client_request"].search(text):
        return "client_request"
    return "routine_or_unclear"


# -------------------------------------------------------------
# 2. Dynamic Ingestion & Processing (Requirement 4)
# -------------------------------------------------------------
@st.cache_data
def load_and_process_data(uploaded_files=None):
    file_map = {}
    if uploaded_files:
        for f in uploaded_files:
            file_map[f.name] = pl.read_csv(f)

    shifts_raw = file_map.get("shifts.csv") or pl.read_csv("data/shifts.csv")
    emp_raw = file_map.get("employees.csv") or pl.read_csv(
        "data/employees.csv"
    )
    payroll_raw = file_map.get("payroll_details.csv") or pl.read_csv(
        "data/payroll_details.csv"
    )
    notes_raw = file_map.get("shift_notes.csv") or pl.read_csv(
        "data/shift_notes.csv"
    )

    # Classify notes dynamically
    notes_df = notes_raw.with_columns(
        pl.col("note")
        .map_elements(classify_text, return_dtype=pl.String, skip_nulls=False)
        .fill_null("routine_or_unclear")
        .alias("category")
    )

    # Clean shifts & apply role-conditioned median imputation
    shifts = shifts_raw.join(
        emp_raw.select(["employee_id", "role", "shift_pattern"]),
        on="employee_id",
        how="left",
    )
    shifts = (
        shifts.with_columns(
            pl.concat_str(
                [pl.col("shift_date"), pl.col("clock_in_time")], separator=" "
            )
            .str.to_datetime("%Y-%m-%d %H:%M")
            .alias("in_dt"),
            pl.concat_str(
                [pl.col("shift_date"), pl.col("clock_out_time")], separator=" "
            )
            .str.to_datetime("%Y-%m-%d %H:%M")
            .alias("out_dt"),
        )
        .with_columns(
            pl.when(pl.col("out_dt") < pl.col("in_dt"))
            .then(pl.col("out_dt") + pl.duration(days=1))
            .otherwise(pl.col("out_dt"))
            .alias("out_dt")
        )
        .with_columns(
            (
                (pl.col("out_dt") - pl.col("in_dt")).dt.total_minutes() / 60.0
            ).alias("duration_hrs")
        )
        .with_columns(
            pl.when(pl.col("duration_hrs").is_not_null())
            .then(pl.col("duration_hrs"))
            .when(pl.col("role") == "Security Guard")
            .then(pl.lit(12.0))
            .otherwise(pl.lit(9.0))
            .alias("duration_hrs")
        )
    )

    # Detect current week vs history
    shifts = shifts.with_columns(
        pl.col("in_dt")
        .dt.truncate("1w")
        .dt.to_string("%Y-%m-%d")
        .alias("week_starting"),
        pl.col("in_dt").dt.weekday().alias("weekday"),
    )

    latest_week = shifts["week_starting"].max()
    curr_shifts = shifts.filter(pl.col("week_starting") == latest_week)
    hist_shifts = shifts.filter(pl.col("week_starting") < latest_week)

    # Compute empirical weekend pacing per role/shift_pattern from completed history
    role_weekend_pace = (
        hist_shifts.filter(pl.col("weekday") > 3)
        .group_by(["role", "shift_pattern"])
        .agg(
            pl.col("duration_hrs").sum().alias("total_weekend_hrs"),
            pl.col("employee_id").n_unique().alias("num_emps"),
        )
        .with_columns(
            (pl.col("total_weekend_hrs") / (pl.col("num_emps") * 9.0)).alias(
                "avg_thu_sun_hrs"
            )
        )
    )

    # Active week calculations (Mon-Wed)
    curr_hours = curr_shifts.group_by("employee_id").agg(
        pl.col("duration_hrs").sum().alias("hours_logged"),
        pl.col("shift_id").count().alias("shifts_count"),
        pl.col("site_id").first().alias("active_site_id"),
    )

    # Join supervisor notes for active week
    active_notes = (
        curr_shifts.join(notes_df, on="shift_id", how="inner")
        .group_by("employee_id")
        .agg(
            pl.col("category").unique().alias("note_categories"),
            pl.col("note").first().alias("sample_note"),
        )
    )

    # Predict risk & breaches
    summary = (
        emp_raw.join(curr_hours, on="employee_id", how="left")
        .with_columns(
            pl.col("hours_logged").fill_null(0.0),
            pl.col("shifts_count").fill_null(0),
            pl.col("active_site_id").fill_null(pl.col("primary_site_id")),
        )
        .join(
            role_weekend_pace.select(
                ["role", "shift_pattern", "avg_thu_sun_hrs"]
            ),
            on=["role", "shift_pattern"],
            how="left",
        )
        .with_columns(pl.col("avg_thu_sun_hrs").fill_null(10.0))
        .with_columns(
            (pl.col("hours_logged") + pl.col("avg_thu_sun_hrs")).alias(
                "projected_total"
            )
        )
        .with_columns(
            (
                1.0
                / (1.0 + np.exp(-(pl.col("projected_total") - 55.0) / 3.5))
            ).alias("risk_score")
        )
        .with_columns(
            (pl.col("risk_score") >= 0.50).cast(pl.Int64).alias("will_breach")
        )
        .join(
            payroll_raw.select(["employee_id", "hourly_rate"]),
            on="employee_id",
            how="left",
        )
        .join(active_notes, on="employee_id", how="left")
    )

    return summary, notes_df, shifts


# -------------------------------------------------------------
# 3. Sidebar: File Upload for Next Week (Requirement 4)
# -------------------------------------------------------------
st.sidebar.title("Operational Setup")
st.sidebar.markdown("**Zero-Developer Data Ingestion**")
uploaded = st.sidebar.file_uploader(
    "Drop next week's export CSVs here:",
    accept_multiple_files=True,
    type=["csv"],
)

summary_df, notes_df, all_shifts = load_and_process_data(uploaded)

site_options = ["All Sites"] + sorted(
    summary_df["active_site_id"].unique().to_list()
)
selected_site = st.sidebar.selectbox("Filter Site:", site_options)

if selected_site != "All Sites":
    view_df = summary_df.filter(pl.col("active_site_id") == selected_site)
else:
    view_df = summary_df

# -------------------------------------------------------------
# 4. Top-Level Ops KPI Strip
# -------------------------------------------------------------
st.title("Ops Room: Weekly Overtime Triage")
st.caption(
    f"Cutoff: Wednesday 23:59 | Reg Threshold: 45h Ordinary + 10h Overtime Cap (55h Total)"
)

breached_view = view_df.filter(pl.col("will_breach") == 1)
total_breaches = len(breached_view)

# Financial Risk Calculation
# Excess hours above 55 * hourly rate * 1.5 OT penalty
estimated_overtime_liability = (
    breached_view.with_columns(
        ((pl.col("projected_total") - 45.0) * pl.col("hourly_rate") * 1.5).alias(
            "ot_liability"
        )
    )["ot_liability"].sum()
    if total_breaches > 0
    else 0.0
)

col1, col2, col3 = st.columns(3)
col1.metric("Projected Breaches", f"{total_breaches} Staff")
col2.metric(
    "Total OT Pay at Risk", f"R {estimated_overtime_liability:,.2f}"
)
col3.metric(
    "Active Staff Mon–Wed", f"{len(view_df.filter(pl.col('hours_logged') > 0))}"
)

st.divider()

# -------------------------------------------------------------
# 5. Requirement 1 & 2: Who is Breaching & What To Do About It
# -------------------------------------------------------------
st.subheader("Roster Intervention Queue (Prioritized by Risk)")

if total_breaches == 0:
    st.success("No compliance breaches projected for this selection.")
else:
    # Build Available Relief Roster (Staff at same site, same role, <= 24h logged)
    relief_roster = view_df.filter(
        (pl.col("will_breach") == 0) & (pl.col("hours_logged") <= 24.0)
    ).sort("hours_logged")

    for row in (
        breached_view.sort("risk_score", descending=True).to_dicts()
    ):
        site = row["active_site_id"]
        role = row["role"]
        name = row["full_name"]
        emp_id = row["employee_id"]
        cur_h = row["hours_logged"]
        proj_h = row["projected_total"]
        score = row["risk_score"]
        sample_note = row.get("sample_note")
        cats = row.get("note_categories")

        # Find best substitution candidate
        candidates = relief_roster.filter(
            (pl.col("active_site_id") == site) & (pl.col("role") == role)
        ).to_dicts()
        sub_text = (
            f"Stand down from weekend shifts. Reassign to **{candidates[0]['full_name']}** ({candidates[0]['employee_id']}) who only has **{candidates[0]['hours_logged']:.1f}h** logged."
            if candidates
            else f"Stand down from weekend shifts. No internal {role} available at {site}; request cross-site relief from ST-01/ST-02."
        )

        why_text = (
            f"Note logged: *'{sample_note}'* (Category: `{cats[0]}`)"
            if sample_note
            else "Shift clustering: Worker scheduled on tight consecutive rotation without supervisor note."
        )

        with st.expander(
            f"🚨 {name} ({emp_id}) — {role} @ {site} | Risk: {score*100:.0f}%",
            expanded=True,
        ):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write(f"**Current (Mon-Wed):** {cur_h:.1f} hrs")
                st.write(f"**Projected (Sunday):** {proj_h:.1f} hrs")
                st.write(
                    f"**Projected Breach:** +{proj_h - 55.0:.1f} hrs over legal cap"
                )
            with c2:
                st.markdown(f"**Why it happened:** {why_text}")
                st.markdown(f"**Action Required Today:** {sub_text}")

# -------------------------------------------------------------
# 6. Requirement 3: The Two Piles Breakdown
# -------------------------------------------------------------
st.divider()
st.subheader("Shift Notes: Root-Cause Split (The Two Piles)")

classified_shifts = all_shifts.join(
    notes_df, on="shift_id", how="left"
).with_columns(
    pl.when(pl.col("category") == "client_request")
    .then(pl.lit("Client Requested (Billable)"))
    .when(
        pl.col("category").is_in(
            ["relief_no_show", "equipment_failure", "handover_delay"]
        )
    )
    .then(pl.lit("Operational Failure (Unrecoverable)"))
    .otherwise(pl.lit("Routine / No Incident"))
    .alias("pile")
)

pile_counts = (
    classified_shifts.group_by(["site_id", "pile"])
    .len()
    .sort(["site_id", "pile"])
    .to_pandas()
)
st.dataframe(
    pile_counts.pivot(
        index="site_id", columns="pile", values="len"
    ).fillna(0),
    use_container_width=True,
)