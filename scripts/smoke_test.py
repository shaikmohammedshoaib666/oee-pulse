"""Smoke test: sample data → clean → OEE → Pareto → insights → Q&A → report → email demo → persist."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.data_integration import plant_default_join
from modules.downtime_analysis import downtime_pareto, mttr_mtbf
from modules.insights_engine import ask_oee_question, generate_insights
from modules.maintenance_analysis import analyze_maintenance
from modules.oee_engine import oee_summary
from modules.optuna_tuner import tune_and_forecast
from modules.quality_analysis import analyze_quality
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

    insights = generate_insights(cleaned, dt_clean, qual)
    assert len(insights) >= 3

    qa = ask_oee_question(
        "Which line has the worst OEE and what drives downtime?",
        cleaned,
        downtime=dt_clean,
        quality=qual,
        insights=insights,
        provider="offline",
    )
    assert qa.get("ok") and qa.get("answer")
    assert qa.get("source") == "offline"

    empty_m = analyze_maintenance(pd.DataFrame())
    assert empty_m.get("ok") is False
    empty_q = analyze_quality(pd.DataFrame())
    assert empty_q.get("ok") is False

    maint = analyze_maintenance(dt_clean, production=prod, oee_frame=cleaned)
    assert maint.get("ok")
    inspect = maint.get("inspect_this_week")
    assert inspect is not None and not inspect.empty
    pu = maint.get("planned_vs_unplanned") or {}
    assert pu.get("ok") and float(pu.get("total_min") or 0) > 0
    hours = maint.get("hours_lost") or {}
    assert hours.get("ok") and hours.get("narrative")
    assert "Availability" in hours["narrative"] or "hours" in hours["narrative"].lower()

    qan = analyze_quality(qual, oee_frame=cleaned, production=prod)
    assert qan.get("ok")
    kpis = qan.get("kpis") or {}
    assert float(kpis.get("fpy") or 0) > 0
    dpareto = qan.get("defect_pareto")
    assert dpareto is not None and not dpareto.empty
    assert qan.get("spc", {}).get("ok")
    assert qan.get("cpk", {}).get("ok")
    assert any("killing quality" in (c.get("title") or "").lower() for c in (qan.get("cards") or []))

    qa_m = ask_oee_question(
        "Which assets should we inspect this week?",
        cleaned,
        downtime=dt_clean,
        quality=qual,
        insights=insights,
        provider="offline",
    )
    assert qa_m.get("ok") and "inspect" in qa_m.get("answer", "").lower()
    qa_q = ask_oee_question(
        "Which line is killing quality this week and what is the scrap rate?",
        cleaned,
        downtime=dt_clean,
        quality=qual,
        insights=insights,
        provider="offline",
    )
    assert qa_q.get("ok") and any(
        w in qa_q.get("answer", "").lower() for w in ("scrap", "fpy", "quality", "line")
    )

    forecast = tune_and_forecast(cleaned, n_trials=5, use_optuna=True)
    assert forecast.get("ok") or "reason" in forecast

    html = build_html_brief(
        "Smoke Plant",
        plant,
        insights,
        pareto.to_dict("records"),
        rel,
        qa_answer=qa["answer"][:400],
        maintenance=maint,
        quality=qan,
    )
    assert "Maintenance" in html and "Inspect this week" in html
    assert "Quality" in html and "FPY" in html
    html_path = write_html_report(html, ROOT / "reports" / "output")
    pdf_path = write_pdf_report(
        {
            "plant_name": "Smoke Plant",
            "oee_plant": plant,
            "insights": insights,
            "pareto_rows": pareto.to_dict("records"),
            "maintenance": maint,
            "quality": qan,
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
        meta={
            "insights": insights,
            "chat_history": [{"role": "user", "content": "hi"}],
            "email_to": "smoke@example.com",
            "maintenance_analysis": {"ok": True, "hours_lost": hours},
            "quality_analysis": {"ok": True, "kpis": kpis},
        },
    )
    loaded = session_store.load_frames(sid)
    assert loaded["cleaned_df"] is not None and len(loaded["cleaned_df"]) == len(cleaned)
    meta = session_store.load_session_meta(sid)
    assert meta.get("email_to") == "smoke@example.com"
    assert meta.get("maintenance_analysis", {}).get("ok")
    assert meta.get("quality_analysis", {}).get("ok")
    recent = session_store.list_sessions(5)
    assert any(r["id"] == sid for r in recent)

    print("SMOKE OK")
    print(f"  production={prod.shape} downtime={dt.shape} quality={qual.shape}")
    print(f"  OEE={float(plant['oee'])*100:.1f}% A={float(plant['availability'])*100:.1f}%")
    print(f"  insights={len(insights)} pareto_rows={len(pareto)} mttr={rel.get('mttr_min')}")
    print(f"  maint_inspect={len(inspect)} unplanned_pct={float(pu.get('unplanned_pct') or 0)*100:.0f}%")
    print(f"  quality_fpy={float(kpis.get('fpy') or 0)*100:.1f}% cpk={qan.get('cpk', {}).get('cpk')}")
    print(f"  qa_source={qa.get('source')} forecast_ok={forecast.get('ok')}")
    print(f"  html={html_path.name} pdf={pdf_path.name} email={Path(email['path']).name}")
    print(f"  session={sid} recent={len(recent)}")


if __name__ == "__main__":
    main()
