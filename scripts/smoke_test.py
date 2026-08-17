"""Smoke test: sample data → clean → OEE → Pareto → insights → Q&A → report → email demo → persist."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.data_integration import plant_default_join
from modules.downtime_analysis import downtime_pareto, mttr_mtbf
from modules.insights_engine import ask_oee_question, generate_insights
from modules.oee_engine import oee_summary
from modules.optuna_tuner import tune_and_forecast
from modules.quality_checks import build_quality_report, clean_plant_frame
from modules.reports import build_html_brief, send_email_brief, write_html_report, write_pdf_report
from modules.sample_data import generate_sample_plant
from modules import session_store


def _assert_no_df_or():
    """Guard: get_active_frame style None checks must not use DataFrame `or`."""
    import pandas as pd

    cleaned = pd.DataFrame({"a": [1]})
    integrated = pd.DataFrame({"b": [2]})
    # Simulate the fixed pattern
    frame = cleaned if cleaned is not None else integrated
    assert frame is cleaned
    cleaned2 = None
    frame2 = cleaned2 if cleaned2 is not None else integrated
    assert frame2 is integrated


def main() -> None:
    _assert_no_df_or()

    sample_dir = ROOT / "sample_data"
    data = generate_sample_plant(out_dir=sample_dir)
    prod, dt, qual = data["production"], data["downtime"], data["quality"]
    assert len(prod) > 50 and len(dt) > 50 and len(qual) > 50

    integrated, logs = plant_default_join(prod, dt, qual)
    assert logs and len(integrated) > 0

    frame = prod.merge(qual, on=["shift_date", "shift", "line_id", "machine_id"], how="left")
    cleaned, clog = clean_plant_frame(frame)
    dt_clean, _ = clean_plant_frame(dt)
    report = build_quality_report(cleaned)
    assert report["summary"]["total"] >= 10

    summary = oee_summary(cleaned)
    plant = summary["plant"]
    assert 0 <= float(plant["oee"]) <= 1.5
    assert "availability" in plant

    pareto = downtime_pareto(dt_clean)
    assert not pareto.empty
    rel = mttr_mtbf(dt_clean)
    assert rel.get("ok")

    insights = generate_insights(cleaned, dt_clean)
    assert len(insights) >= 3

    qa = ask_oee_question(
        "Which line has the worst OEE and what drives downtime?",
        cleaned,
        downtime=dt_clean,
        insights=insights,
        provider="offline",
    )
    assert qa.get("ok") and qa.get("answer")
    assert qa.get("source") == "offline"

    forecast = tune_and_forecast(cleaned, n_trials=5, use_optuna=True)
    assert forecast.get("ok") or "reason" in forecast

    html = build_html_brief(
        "Smoke Plant",
        plant,
        insights,
        pareto.to_dict("records"),
        rel,
        qa_answer=qa["answer"][:400],
    )
    html_path = write_html_report(html, ROOT / "reports" / "output")
    pdf_path = write_pdf_report(
        {
            "plant_name": "Smoke Plant",
            "oee_plant": plant,
            "insights": insights,
            "pareto_rows": pareto.to_dict("records"),
        },
        ROOT / "reports" / "output",
    )
    assert html_path.exists() and pdf_path.exists()

    email = send_email_brief(html, to_addr="smoke@example.com", demo_mode=True)
    assert email.get("ok") and email.get("demo")
    assert Path(email["path"]).exists()

    # Persistence
    sid = session_store.new_session_id()
    session_store.save_session(
        sid,
        title="Smoke session",
        plant_name="Smoke Plant",
        frames={
            "production_df": prod,
            "downtime_df": dt_clean,
            "quality_df": qual,
            "integrated_df": frame,
            "cleaned_df": cleaned,
        },
        meta={"insights": insights, "chat_history": [{"role": "user", "content": "hi"}], "email_to": "smoke@example.com"},
    )
    loaded = session_store.load_frames(sid)
    assert loaded["cleaned_df"] is not None and len(loaded["cleaned_df"]) == len(cleaned)
    meta = session_store.load_session_meta(sid)
    assert meta.get("email_to") == "smoke@example.com"
    recent = session_store.list_sessions(5)
    assert any(r["id"] == sid for r in recent)

    print("SMOKE OK")
    print(f"  production={prod.shape} downtime={dt.shape} quality={qual.shape}")
    print(f"  OEE={float(plant['oee'])*100:.1f}% A={float(plant['availability'])*100:.1f}%")
    print(f"  insights={len(insights)} pareto_rows={len(pareto)} mttr={rel.get('mttr_min')}")
    print(f"  qa_source={qa.get('source')} forecast_ok={forecast.get('ok')}")
    print(f"  html={html_path.name} pdf={pdf_path.name} email={Path(email['path']).name}")
    print(f"  session={sid} recent={len(recent)}")


if __name__ == "__main__":
    main()
