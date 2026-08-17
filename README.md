# OEE Pulse

**Industry 4.0 analytics SaaS** for plant managers and industrial engineers — focused on **OEE**, **unplanned downtime**, and actionable production insights.

Upload messy plant CSVs → clean & quality-check → compute OEE → Pareto downtime → manager-language insights → HTML/PDF/email brief.

![Python](https://img.shields.io/badge/python-3.9%2B-slate)
![Streamlit](https://img.shields.io/badge/streamlit-SaaS%20MVP-amber)

## Product vision

- Reduce unplanned downtime
- Manage OEE losses (Availability / Performance / Quality)
- Optimize maintenance windows and production planning
- Demo-ready synthetic plant data out of the box

## Features

| Module | What it does |
|--------|----------------|
| **Upload & Integrate** | Multi-file upload (production, downtime, quality) + SQL-style 3-table joins (pandas / optional DuckDB) |
| **Clean & Quality** | Industrial cleaning + statistical / ML quality checks (z-score, IQR, Isolation Forest, DBSCAN, PCA drift, OEE domain rules) |
| **OEE Cockpit** | Availability × Performance × Quality, plant / line / machine / shift breakdowns, loss decomposition |
| **Downtime Analysis** | Pareto of downtime codes, MTTR / MTBF lite, chronic machines |
| **Insights** | Manager language (“Line 3 lost X% availability to changeovers”) + optional Optuna-tuned scrap/downtime forecast |
| **Reports** | HTML + PDF export, email brief (demo mode by default) |

## Quick start (local)

```bash
cd oee-pulse
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m modules.sample_data   # writes sample_data/*.csv
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), then click **Load sample plant data** in the sidebar.

## Streamlit Community Cloud

1. Push this repo to GitHub (already set up if you cloned from the public repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch `main`, main file `app.py`.
4. Deploy. Optional: add secrets from `.env.example` in the Cloud secrets UI.

## Configuration

Copy `.env.example` → `.env`:

```bash
cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `OEE_PULSE_DEMO_MODE` | `true` = simulate email (default) |
| `SMTP_*` | Real SMTP when demo mode is off |
| `PLANT_NAME` | Shown in sidebar / reports |

## Architecture

```
app.py                 Streamlit SaaS shell (sidebar nav + pages)
ui/                    Theme (slate/steel/amber) + session state
modules/
  data_integration.py  Multi-file load + chained joins
  quality_checks.py    Industrial clean + quality report
  oee_engine.py        OEE math + aggregations
  downtime_analysis.py Pareto, MTTR/MTBF
  insights_engine.py   Manager-language insights
  optuna_tuner.py      Optional forecast / anomaly tuning
  reports.py           HTML / PDF / email brief
  sample_data.py       Synthetic multi-line plant CSVs
sample_data/           Generated demo CSVs
reports/output/        Exported briefs
```

**Data flow:** Production + Downtime + Quality → join → clean/QC → OEE engine → Pareto & reliability → insights / Optuna → report.

## Tech stack

Python 3.9+ · Streamlit · pandas · numpy · scikit-learn · Plotly · Optuna · DuckDB (optional) · fpdf2

## License

MIT — built as a demo-ready Industry 4.0 SaaS MVP.
