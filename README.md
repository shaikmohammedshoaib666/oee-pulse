# OEE Pulse

**Industry 4.0 analytics SaaS** for plant managers and industrial engineers — focused on **OEE**, **unplanned downtime**, and actionable production insights.

Upload messy plant CSVs → clean & quality-check → compute OEE → Pareto downtime → manager-language insights + Q&A → HTML/PDF/email brief. Sessions persist across browser reloads.

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
| **Upload & Integrate** | Multi-file upload (production, downtime, quality) + SQL-style 3-table joins (pandas / optional DuckDB). **Saved column mapping** (messy / SAP-like headers → canonical fields, reused next upload). **SAP-style extract templates** (PP/PM/QM CSV + README) |
| **Clean & Quality** | Industrial cleaning + statistical / ML quality checks |
| **OEE Cockpit** | Availability × Performance × Quality, plant / line / machine / shift breakdowns, **$ impact of lost hours**, **Availability hours \| $ lost \| PdM risk** |
| **Downtime Analysis** | Pareto of downtime codes (minutes **and $**), MTTR / MTBF lite, chronic machines |
| **Maintenance** | Reliability overlay: asset risk, planned vs unplanned, inspect-this-week ranking, Availability hours lost beside **$ and PdM risk** |
| **Quality** | SPC / scrap overlay: FPY, defect Pareto, I-MR control limits, Cp/Cpk lite, killer-line cards |
| **Insights** | Manager language + Optuna forecast + LLM/offline Q&A (Gemini/OpenAI), including **inspect this week using $ and risk** |
| **Reports** | HTML + PDF export, email brief (SMTP or demo disk save) including maint, quality, and **lost-OEE $** |
| **Persistence** | SQLite + CSV session store — recent sessions in sidebar, reload last session, **column mappings per plant** |

## Quick start (local)

```bash
cd oee-pulse
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m modules.sample_data   # writes sample_data/*.csv
streamlit run app.py
```

Open the URL Streamlit prints, then click **Load sample plant data** in the sidebar.

Smoke test:

```bash
python scripts/smoke_test.py
```

## Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch `main`, main file `app.py`.
4. Add secrets (see below) in **Settings → Secrets**.
5. Redeploy after pushes to `main`.

## Secrets / `.env`

Copy `.env.example` → `.env`, or paste into Streamlit Cloud secrets:

```toml
OEE_PULSE_DEMO_MODE = "true"
EMAIL_TO = "you@company.com"
PLANT_NAME = "North Plant"

# Real email (set DEMO false + SMTP)
# OEE_PULSE_DEMO_MODE = "false"
# SMTP_HOST = "smtp.gmail.com"
# SMTP_PORT = "587"
# SMTP_USER = "you@gmail.com"
# SMTP_PASSWORD = "app-password"
# SMTP_FROM = "you@gmail.com"
# SMTP_TO = "plant.manager@company.com"

# Optional LLM for Insights / Reports Q&A
# GEMINI_API_KEY = "..."
# GEMINI_MODEL = "gemini-2.0-flash"
# OPENAI_API_KEY = "sk-..."
# OPENAI_MODEL = "gpt-4o-mini"
# AI_DEFAULT_PROVIDER = "gemini"
```

| Variable | Purpose |
|----------|---------|
| `OEE_PULSE_DEMO_MODE` | `true` = save email HTML under `reports/output/email_demo/` |
| `EMAIL_TO` / `SMTP_TO` | Default report recipient |
| `SMTP_*` | Real SMTP when demo mode is off |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | LLM Q&A (offline fallback always works) |
| `PLANT_NAME` | Sidebar / reports / saved column-mapping key |

**In-app (not secrets):** sidebar **Lost-hour $ rates** — plant default `$ / hour`, optional `$ / good unit`, per-line rates, and a **Rates match finance** checkbox. Until that box is checked, dollar figures are labeled **management estimate**. Sample plant data ships with synthetic line rates (L1 $850 / L2 $1100 / L3 $720 per hour) and vibration / temp / current columns.

## Architecture

```
app.py                 Streamlit SaaS shell (sidebar nav + pages)
ui/                    Theme + session helpers (get_active_frame, persist)
modules/
  config_secrets.py    Env + Streamlit secrets
  session_store.py     SQLite metadata + CSV frames under data/sessions/
  data_integration.py  Multi-file load + chained joins
  quality_checks.py    Industrial clean + quality report
  oee_engine.py        OEE math + aggregations
  column_mapping.py    Saved SAP/messy → canonical field mapping
  sap_templates.py     PP/PM/QM extract CSV templates + README ZIP
  finance_impact.py    $ / lost hour, Pareto in $, hours vs PdM risk
  downtime_analysis.py Pareto, MTTR/MTBF
  maintenance_analysis.py Reliability risk, planned vs unplanned, inspect-this-week
  quality_analysis.py  Scrap/FPY, defect Pareto, SPC lite, Cp/Cpk
  insights_engine.py   Manager insights + ask_oee_question (LLM/offline)
  optuna_tuner.py      Optional forecast / anomaly tuning
  reports.py           HTML / PDF / email brief
  sample_data.py       Synthetic multi-line plant CSVs
```

**Data flow:** Production + Downtime + Quality → column mapping (saved) → join → clean/QC → OEE cockpit ($ + PdM risk) → downtime / maintenance / quality overlays → insights / Q&A → report / email.

## Email automation MVP

- **Email report now** sends (or demo-saves) the latest brief to `EMAIL_TO`.
- Streamlit Cloud does not run long-lived schedulers; use external cron / GitHub Actions later, or click daily.
- Inbound email → auto-report is a **future connector** (documented in-app).

## License

MIT — built as a demo-ready Industry 4.0 SaaS MVP.
