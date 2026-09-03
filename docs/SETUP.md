# Setup Guide

A step-by-step operational guide to configure the Python environment, generate both required CSV deliverables (`note_classifications.csv` and `predictions.csv`), and launch the Streamlit dashboard on port 3000.

---

```bash
uv sync
```


<!-- ### Generate Deliverable 1: `note_classifications.csv`

The brief specifies exact columns: `shift_id,category,note`.

Run this command in your terminal to classify supervisor notes into the taxonomy (`relief_no_show`, `client_request`, `equipment_failure`, `handover_delay`, `routine_or_unclear`) and export `note_classifications.csv`:

```bash
python scripts/classify_notes.py
```

*Direct inline generator if you haven't saved the script:*

```bash
python -c "import polars as pl, re; patterns={'relief_no_show': re.compile(r'(not?\s+pitch|didn\'?t\s+come|no[- ]show|stood\s+in|covered|replacement|relie[fv]|akafikanga|akezanga|ngimele|afwesig|geen\s+aflos|double\s+shift)', re.I), 'equipment_failure': re.compile(r'(generator|masjien|stukkend|broken|fault|power\s+outage|load[\s-]?shedding|gate\s+fault|alarm|water\s+leak|herstel|krag)', re.I), 'handover_delay': re.compile(r'(handover|oorhandiging|keys?|sleutel|lock\s+box|missing\s+keys)', re.I), 'client_request': re.compile(r'(client\s+(asked|wanted|requested|signed|email)|approved|customer\s+request|klient|gemagtig|extra\s+(patrol|man|guard|hours|shift)|stocktake)', re.I)}; routine=re.compile(r'(all\s+good|quiet\s+shift|sharp|ok|normal\s+shift|alles\s+reg|lungile|routine|nothing\s+to\s+report)', re.I); classify = lambda t: 'routine_or_unclear' if not t or not str(t).strip() or (routine.search(str(t).strip()) and not any(p.search(str(t).strip()) for p in patterns.values())) else next((k for k, p in patterns.items() if p.search(str(t).strip())), 'routine_or_unclear'); df = pl.read_csv('data/shift_notes.csv').with_columns(pl.col('note').map_elements(classify, return_dtype=pl.String, skip_nulls=False).fill_null('routine_or_unclear').alias('category')).select(['shift_id', 'category', 'note']); df.write_csv('note_classifications.csv'); print(f'Exported note_classifications.csv ({len(df)} rows)')"
```

<!-- ### Generate Deliverable 2: `predictions.csv`

The brief specifies exact columns: `employee_id,will_breach,risk_score` for all 213 employees in the target week (10–16 August 2026).

Run your prediction script:

```bash
python scripts/predict_overtime.py
```

*Direct inline generator if you need to build it immediately:*

```bash
python -c "import polars as pl, numpy as np; shifts=pl.read_csv('data/shifts.csv'); emp=pl.read_csv('data/employees.csv'); s=shifts.join(emp.select(['employee_id', 'role']), on='employee_id').with_columns(pl.concat_str(['shift_date', 'clock_in_time'], separator=' ').str.to_datetime('%Y-%m-%d %H:%M').alias('i'), pl.concat_str(['shift_date', 'clock_out_time'], separator=' ').str.to_datetime('%Y-%m-%d %H:%M').alias('o')).with_columns(pl.when(pl.col('o') < pl.col('i')).then(pl.col('o') + pl.duration(days=1)).otherwise(pl.col('o')).alias('o')).with_columns(((pl.col('o') - pl.col('i')).dt.total_minutes() / 60.0).alias('d')).with_columns(pl.when(pl.col('d').is_not_null()).then(pl.col('d')).when(pl.col('role') == 'Security Guard').then(pl.lit(12.0)).otherwise(pl.lit(9.0)).alias('d')).with_columns(pl.col('i').dt.truncate('1w').dt.to_string('%Y-%m-%d').alias('w'), pl.col('i').dt.weekday().alias('day')); h_p = s.filter(pl.col('w') < '2026-08-10', pl.col('day') > 3).group_by('role').agg((pl.col('d').sum() / (pl.col('employee_id').n_unique() * 9.0)).alias('exp_wknd')); curr = s.filter(pl.col('w') == '2026-08-10').group_by('employee_id').agg(pl.col('d').sum().alias('mon_wed')); res = emp.select(['employee_id', 'role']).join(curr, on='employee_id', how='left').with_columns(pl.col('mon_wed').fill_null(0.0)).join(h_p, on='role', how='left').with_columns(pl.col('exp_wknd').fill_null(10.0)).with_columns((pl.col('mon_wed') + pl.col('exp_wknd')).alias('proj')).with_columns((1.0 / (1.0 + np.exp(-(pl.col('proj') - 55.0) / 3.5))).round(2).alias('risk_score')).with_columns((pl.col('risk_score') >= 0.50).cast(pl.Int64).alias('will_breach')).select(['employee_id', 'will_breach', 'risk_score']); res.write_csv('predictions.csv'); print(f'Exported predictions.csv ({len(res)} rows, {res[\"will_breach\"].sum()} breaches)')"
```

<!-- ### Verify Repo Output Files

Confirm that both root deliverable files exist and have exact headers:

```bash
# Check note_classifications.csv header and row count
python -c "import polars as pl; df = pl.read_csv('note_classifications.csv'); print('Notes:', df.columns, len(df))"

# Check predictions.csv header and row count
python -c "import polars as pl; df = pl.read_csv('predictions.csv'); print('Preds:', df.columns, len(df))"
```

Expected terminal output:

```text
Notes: ['shift_id', 'category', 'note'] 2117
Preds: ['employee_id', 'will_breach', 'risk_score'] 213
```

--- -->

### Launch the Local Ops Room Dashboard

Run the Streamlit application on port 3000:

```bash
streamlit run app.py --server.port 3000
```

Open `http://localhost:3000` in your browser. Use your browser's responsive mobile toggle (`Ctrl + Shift + M` / `Cmd + Shift + M`) to simulate the phone viewport used by contract managers in the field.

