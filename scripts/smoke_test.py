"""Smoke test: sample data → clean → OEE → Pareto → insights → report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.data_integration import plant_default_join
from modules.downtime_analysis import downtime_pareto, mttr_mtbf
from modules.insights_engine import generate_insights
from modules.oee_engine import oee_summary
from modules.optuna_tuner import tune_and_forecast
from modules.quality_checks import build_quality_report, clean_plant_frame
from modules.reports import build_html_brief, write_html_report, write_pdf_report
from modules.sample_data import generate_sample_plant


def main() -> None:
    sample_dir = ROOT / "sample_data"
    data = generate_sample_plant(out_dir=sample_dir)
    prod, dt, qual = data["production"], data["downtime"], data["quality"]
    assert len(prod) > 50 and len(dt) > 50 and len(qual) > 50

    integrated, logs = plant_default_join(prod, dt, qual)
    assert logs and len(integrated) > 0

    # Shift-level frame for OEE
    frame = prod.merge(qual, on=["shift_date", "shift", "line_id", "machine_id"], how="left")
    cleaned, clog = clean_plant_frame(frame)
    report = build_quality_report(cleaned)
    assert report["summary"]["total"] >= 10

    summary = oee_summary(cleaned)
    plant = summary["plant"]
    assert 0 <= float(plant["oee"]) <= 1.5
    assert "availability" in plant

    pareto = downtime_pareto(dt)
    assert not pareto.empty
    rel = mttr_mtbf(dt)
    assert rel.get("ok")

    insights = generate_insights(cleaned, dt)
    assert len(insights) >= 3

    forecast = tune_and_forecast(cleaned, n_trials=5, use_optuna=True)
    assert forecast.get("ok") or "reason" in forecast

    html = build_html_brief(
        "Smoke Plant",
        plant,
        insights,
        pareto.to_dict("records"),
        rel,
    )
    html_path = write_html_report(html, ROOT / "reports" / "output")
    pdf_path = write_pdf_report(
        {"plant_name": "Smoke Plant", "oee_plant": plant, "insights": insights, "pareto_rows": pareto.to_dict("records")},
        ROOT / "reports" / "output",
    )
    assert html_path.exists() and pdf_path.exists()

    print("SMOKE OK")
    print(f"  production={prod.shape} downtime={dt.shape} quality={qual.shape}")
    print(f"  OEE={float(plant['oee'])*100:.1f}% A={float(plant['availability'])*100:.1f}%")
    print(f"  insights={len(insights)} pareto_rows={len(pareto)} mttr={rel.get('mttr_min')}")
    print(f"  forecast_ok={forecast.get('ok')} optuna={forecast.get('optuna_used')}")
    print(f"  html={html_path.name} pdf={pdf_path.name}")


if __name__ == "__main__":
    main()
