"""Manager-language business insights + LLM / offline Q&A for OEE Pulse."""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from modules.config_secrets import (
    get_ai_default_provider,
    get_gemini_api_key,
    get_gemini_model,
    get_openai_api_key,
    get_openai_model,
    llm_status,
)
from modules.downtime_analysis import downtime_pareto, mttr_mtbf, top_chronic_machines
from modules.maintenance_analysis import analyze_maintenance, frame_ok as maint_frame_ok
from modules.oee_engine import oee_summary
from modules.quality_analysis import analyze_quality, frame_ok as quality_frame_ok
from modules.quality_checks import find_col

SYSTEM_PROMPT = (
    "You are OEE Pulse, an Industry 4.0 co-pilot for plant managers. "
    "Answer using the provided OEE, downtime Pareto, MTTR/MTBF, maintenance risk, "
    "$ impact of lost hours (management estimate unless rates match finance), "
    "quality scrap/FPY/SPC, and line/machine metrics. "
    "When asked what to inspect this week, use BOTH dollar loss and PdM/failure risk. "
    "Use clear manager language. Be concise, actionable, and grounded in the numbers. "
    "Do not invent machines or percentages that are not in the context."
)


def generate_insights(
    production_oee: pd.DataFrame,
    downtime: pd.DataFrame | None = None,
    quality: pd.DataFrame | None = None,
    finance_rates: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return prioritized insights in plant-manager language."""
    insights: list[dict[str, Any]] = []
    summary = oee_summary(production_oee)
    plant = summary["plant"]
    oee = float(plant.get("oee", 0))
    avail = float(plant.get("availability", 0))
    perf = float(plant.get("performance", 0))
    qual = float(plant.get("quality", 0))

    insights.append(
        {
            "priority": "high" if oee < 0.65 else ("medium" if oee < 0.85 else "low"),
            "title": "Plant OEE position",
            "message": (
                f"Plant OEE is {oee*100:.1f}% "
                f"(A {avail*100:.1f}% × P {perf*100:.1f}% × Q {qual*100:.1f}%). "
                f"Gap to world-class 85%: {summary['world_class_gap']*100:.1f} pts."
            ),
        }
    )

    losses = {
        "Availability": float(plant.get("availability", 0)),
        "Performance": float(plant.get("performance", 0)),
        "Quality": float(plant.get("quality", 0)),
    }
    worst = min(losses, key=losses.get)
    insights.append(
        {
            "priority": "high",
            "title": f"Primary loss pillar: {worst}",
            "message": (
                f"{worst} is the weakest OEE lever at {losses[worst]*100:.1f}%. "
                f"Focus the next kaizen wave here before spreading effort across all three."
            ),
        }
    )

    by_line = summary["by_line"]
    if isinstance(by_line, pd.DataFrame) and not by_line.empty and "line_id" in by_line.columns:
        worst_line = by_line.sort_values("oee").iloc[0]
        best_line = by_line.sort_values("oee").iloc[-1]
        avail_loss_pct = (1 - float(worst_line["availability"])) * 100
        insights.append(
            {
                "priority": "high",
                "title": f"Line {worst_line['line_id']} underperforming",
                "message": (
                    f"Line {worst_line['line_id']} OEE is {float(worst_line['oee'])*100:.1f}% "
                    f"vs Line {best_line['line_id']} at {float(best_line['oee'])*100:.1f}%. "
                    f"Line {worst_line['line_id']} lost {avail_loss_pct:.1f}% availability — "
                    f"review changeovers and unplanned stops first."
                ),
            }
        )

    by_machine = summary["by_machine"]
    if isinstance(by_machine, pd.DataFrame) and not by_machine.empty:
        m = by_machine.sort_values("oee").iloc[0]
        insights.append(
            {
                "priority": "medium",
                "title": f"Machine {m['machine_id']} is the bottleneck",
                "message": (
                    f"Machine {m['machine_id']} posts OEE {float(m['oee'])*100:.1f}% with "
                    f"{float(m.get('downtime_minutes', 0)):.0f} min downtime in scope. "
                    f"Prioritize maintenance window and operator standard work here."
                ),
            }
        )

    by_shift = summary["by_shift"]
    if isinstance(by_shift, pd.DataFrame) and not by_shift.empty and len(by_shift) > 1:
        s_worst = by_shift.sort_values("oee").iloc[0]
        s_best = by_shift.sort_values("oee").iloc[-1]
        insights.append(
            {
                "priority": "medium",
                "title": "Shift performance gap",
                "message": (
                    f"Shift {s_worst['shift']} OEE {float(s_worst['oee'])*100:.1f}% vs "
                    f"Shift {s_best['shift']} at {float(s_best['oee'])*100:.1f}%. "
                    f"Audit changeover discipline and handoff quality on the weaker shift."
                ),
            }
        )

    dt = downtime if downtime is not None else production_oee
    try:
        pareto = downtime_pareto(dt, top_n=5)
        if not pareto.empty:
            top = pareto.iloc[0]
            insights.append(
                {
                    "priority": "high",
                    "title": "Top downtime driver",
                    "message": (
                        f"'{top['cause']}' accounts for {float(top['pct'])*100:.1f}% of downtime "
                        f"({float(top['downtime_minutes']):.0f} min, {int(top['events'])} events). "
                        f"Cumulative top causes reach "
                        f"{float(pareto['cum_pct'].iloc[min(2, len(pareto)-1)])*100:.0f}% "
                        f"by the 3rd cause — classic Pareto opportunity."
                    ),
                }
            )
            change = pareto[pareto["cause"].astype(str).str.lower().str.contains("change")]
            if not change.empty:
                row = change.iloc[0]
                insights.append(
                    {
                        "priority": "high",
                        "title": "Changeover availability loss",
                        "message": (
                            f"Changeovers ('{row['cause']}') consumed "
                            f"{float(row['downtime_minutes']):.0f} minutes "
                            f"({float(row['pct'])*100:.1f}% of downtime). "
                            f"SMED / parallel setup can reclaim availability without CapEx."
                        ),
                    }
                )
    except Exception:
        pass

    reliability = mttr_mtbf(dt)
    if reliability.get("ok"):
        msg = f"MTTR ≈ {reliability['mttr_min']:.1f} min across {reliability['events']} events."
        if "mtbf_min" in reliability:
            msg += f" MTBF ≈ {reliability['mtbf_min']:.1f} min."
            if reliability["mttr_min"] > 0 and reliability.get("mtbf_min", 0) > 0:
                avail_proxy = reliability["mtbf_min"] / (
                    reliability["mtbf_min"] + reliability["mttr_min"]
                )
                msg += f" Reliability availability proxy ≈ {avail_proxy*100:.1f}%."
        insights.append({"priority": "medium", "title": "Reliability snapshot", "message": msg})

    chronic = top_chronic_machines(dt, top_n=3)
    if not chronic.empty:
        mid = find_col(chronic, "machine_id", "machine") or chronic.columns[0]
        names = ", ".join(chronic[mid].astype(str).tolist())
        insights.append(
            {
                "priority": "medium",
                "title": "Chronic downtime machines",
                "message": (
                    f"Highest downtime concentration on {names}. "
                    f"Bundle these into the next planned maintenance window."
                ),
            }
        )

    if oee < 0.75:
        insights.append(
            {
                "priority": "medium",
                "title": "Production planning caution",
                "message": (
                    "With OEE below 75%, master schedule stretch is risky. "
                    "Protect buffer for unplanned stops or defer non-critical SKUs until "
                    "availability recovers above 80%."
                ),
            }
        )

    try:
        from modules.finance_impact import (
            asset_cost_risk_table,
            inspect_this_week_narrative,
            plant_dollar_impact,
        )

        impact = plant_dollar_impact(production_oee, downtime, finance_rates)
        if impact.get("ok"):
            insights.append(
                {
                    "priority": "high",
                    "title": f"Lost OEE dollars ({impact.get('label')})",
                    "message": (
                        f"Availability loss is {float(impact.get('availability_hours') or 0):.1f} hours "
                        f"(${float(impact.get('availability_usd') or 0):,.0f}). "
                        f"Including speed/quality hour-equivalents, estimated impact is "
                        f"${float(impact.get('total_usd') or 0):,.0f} at "
                        f"${float(impact.get('plant_usd_per_hour') or 0):,.0f}/hour. "
                        f"{impact.get('disclaimer')}"
                    ),
                }
            )
        matrix, _ = asset_cost_risk_table(
            downtime, oee_frame=production_oee, production=production_oee, rates=finance_rates
        )
        if isinstance(matrix, pd.DataFrame) and not matrix.empty:
            insights.append(
                {
                    "priority": "high",
                    "title": "Inspect this week ($ and PdM risk)",
                    "message": inspect_this_week_narrative(matrix, finance_rates),
                }
            )
    except Exception:
        pass

    if maint_frame_ok(downtime):
        try:
            maint = analyze_maintenance(downtime, production=production_oee, oee_frame=production_oee)
            for card in maint.get("cards") or []:
                insights.append(card)
        except Exception:
            pass

    if quality_frame_ok(quality) or quality_frame_ok(production_oee):
        try:
            qsrc = quality if quality_frame_ok(quality) else production_oee
            qan = analyze_quality(qsrc, oee_frame=production_oee)
            for card in qan.get("cards") or []:
                insights.append(card)
        except Exception:
            pass

    order = {"high": 0, "medium": 1, "low": 2}
    insights.sort(key=lambda x: order.get(x["priority"], 9))
    return insights


def build_metrics_context(
    frame: pd.DataFrame,
    downtime: pd.DataFrame | None = None,
    insights: list[dict[str, Any]] | None = None,
    quality: pd.DataFrame | None = None,
    finance_rates: dict[str, Any] | None = None,
) -> str:
    """Compact grounded context for LLM / offline answers."""
    summary = oee_summary(frame)
    plant = summary["plant"]
    lines = [
        f"Plant OEE={float(plant.get('oee', 0))*100:.1f}%",
        f"Availability={float(plant.get('availability', 0))*100:.1f}%",
        f"Performance={float(plant.get('performance', 0))*100:.1f}%",
        f"Quality={float(plant.get('quality', 0))*100:.1f}%",
        f"World-class gap={summary['world_class_gap']*100:.1f} pts",
    ]
    by_line = summary["by_line"]
    if isinstance(by_line, pd.DataFrame) and not by_line.empty:
        for _, r in by_line.head(8).iterrows():
            lines.append(
                f"Line {r.get('line_id')}: OEE={float(r.get('oee', 0))*100:.1f}% "
                f"A={float(r.get('availability', 0))*100:.1f}%"
            )
    by_machine = summary["by_machine"]
    if isinstance(by_machine, pd.DataFrame) and not by_machine.empty:
        for _, r in by_machine.sort_values("oee").head(5).iterrows():
            lines.append(
                f"Machine {r.get('machine_id')}: OEE={float(r.get('oee', 0))*100:.1f}% "
                f"downtime={float(r.get('downtime_minutes', 0)):.0f} min"
            )

    dt = downtime if downtime is not None else frame
    try:
        pareto = downtime_pareto(dt, top_n=6)
        for _, r in pareto.iterrows():
            lines.append(
                f"Downtime cause '{r['cause']}': {float(r['downtime_minutes']):.0f} min "
                f"({float(r['pct'])*100:.1f}%)"
            )
    except Exception:
        pass
    rel = mttr_mtbf(dt)
    if rel.get("ok"):
        lines.append(
            f"MTTR={rel.get('mttr_min')} min, MTBF={rel.get('mtbf_min', 'n/a')} min, "
            f"events={rel.get('events')}"
        )
    if insights:
        for i in insights[:6]:
            lines.append(f"Insight [{i.get('priority')}]: {i.get('message')}")

    try:
        from modules.finance_impact import asset_cost_risk_table, plant_dollar_impact

        impact = plant_dollar_impact(frame, downtime, finance_rates)
        if impact.get("ok"):
            lines.append(
                f"$ impact ({impact.get('label')}): Availability ${float(impact.get('availability_usd') or 0):,.0f} "
                f"({float(impact.get('availability_hours') or 0):.1f} h) · "
                f"total ${float(impact.get('total_usd') or 0):,.0f} at "
                f"${float(impact.get('plant_usd_per_hour') or 0):,.0f}/h. "
                f"{impact.get('disclaimer')}"
            )
        matrix, _ = asset_cost_risk_table(
            downtime, oee_frame=frame, production=frame, rates=finance_rates, top_n=6
        )
        if isinstance(matrix, pd.DataFrame) and not matrix.empty:
            for _, r in matrix.head(5).iterrows():
                lines.append(
                    f"Asset {r.get('machine_id')} line {r.get('line_id')}: "
                    f"avail hours lost={float(r.get('availability_hours_lost') or 0):.1f} "
                    f"$lost={float(r.get('usd_lost') or 0):.0f} "
                    f"PdM risk={float(r.get('remaining_risk') or 0):.0f} "
                    f"priority={float(r.get('priority_score') or 0):.0f}"
                )
    except Exception:
        pass

    if maint_frame_ok(downtime):
        try:
            maint = analyze_maintenance(downtime, production=frame, oee_frame=frame)
            hours = maint.get("hours_lost") or {}
            if hours.get("narrative"):
                lines.append(hours["narrative"])
            inspect = maint.get("inspect_this_week")
            if isinstance(inspect, pd.DataFrame) and not inspect.empty:
                for _, r in inspect.head(4).iterrows():
                    lines.append(
                        f"Inspect {r.get('machine_id')} (line {r.get('line_id')}): "
                        f"risk={float(r.get('remaining_risk', 0)):.0f} {r.get('why')}"
                    )
            pu = maint.get("planned_vs_unplanned") or {}
            if pu.get("ok"):
                lines.append(
                    f"Planned downtime={pu.get('planned_min')} min "
                    f"({float(pu.get('planned_pct') or 0)*100:.0f}%), "
                    f"unplanned={pu.get('unplanned_min')} min"
                )
        except Exception:
            pass

    qsrc = quality if quality_frame_ok(quality) else frame
    if quality_frame_ok(qsrc):
        try:
            qan = analyze_quality(qsrc, oee_frame=frame)
            k = qan.get("kpis") or {}
            if k:
                lines.append(
                    f"Scrap={float(k.get('scrap_rate') or 0)*100:.1f}% "
                    f"FPY={float(k.get('fpy') or 0)*100:.1f}% "
                    f"Quality OEE={float(k.get('quality_oee') or 0)*100:.1f}%"
                )
            pareto = qan.get("defect_pareto")
            if isinstance(pareto, pd.DataFrame) and not pareto.empty:
                for _, r in pareto.head(4).iterrows():
                    lines.append(
                        f"Defect '{r.get('defect_code')}': {float(r.get('rejects', 0)):.0f} rejects "
                        f"({float(r.get('pct', 0))*100:.1f}%)"
                    )
            cpk = qan.get("cpk") or {}
            if cpk.get("ok"):
                lines.append(f"Cp={cpk.get('cp')} Cpk={cpk.get('cpk')} on {cpk.get('measurement')}")
        except Exception:
            pass

    return "\n".join(lines)


def _offline_answer(
    question: str,
    frame: pd.DataFrame,
    downtime: pd.DataFrame | None,
    insights: list[dict[str, Any]] | None,
    quality: pd.DataFrame | None = None,
    finance_rates: dict[str, Any] | None = None,
) -> str:
    q = (question or "").lower()
    summary = oee_summary(frame)
    plant = summary["plant"]
    insight_list = insights or generate_insights(frame, downtime, quality, finance_rates)

    if any(w in q for w in ("oee", "overall equipment", "how are we", "plant status", "summary")):
        return (
            f"Plant OEE is {float(plant.get('oee', 0))*100:.1f}% "
            f"(A {float(plant.get('availability', 0))*100:.1f}% × "
            f"P {float(plant.get('performance', 0))*100:.1f}% × "
            f"Q {float(plant.get('quality', 0))*100:.1f}%). "
            f"Gap to 85% world-class: {summary['world_class_gap']*100:.1f} pts. "
            f"Top focus: {insight_list[0]['message'] if insight_list else 'improve weakest pillar.'}"
        )

    quality_intent = any(
        w in q
        for w in (
            "scrap",
            "reject",
            "fpy",
            "first-pass",
            "first pass",
            "yield",
            "defect",
            "cpk",
            "spc",
            "control limit",
            "killing quality",
        )
    ) or ("quality" in q and any(w in q for w in ("line", "worst", "which", "spc", "scrap")))
    maint_intent = any(
        w in q
        for w in (
            "inspect",
            "remaining risk",
            "unplanned",
            "planned vs",
            "hours lost",
            "availability loss",
            "maintenance",
            "this week",
            "failure risk",
            "pdm",
            "dollar",
            "$",
            "costly",
        )
    )
    money_intent = any(
        w in q
        for w in (
            "dollar",
            "$",
            "cost",
            "usd",
            "lost hour",
            "lost oee",
            "finance",
            "money",
            "impact",
        )
    )

    if quality_intent:
        qsrc = quality if quality_frame_ok(quality) else frame
        try:
            qan = analyze_quality(qsrc, oee_frame=frame)
            k = qan.get("kpis") or {}
            if any(w in q for w in ("scrap", "fpy", "yield", "reject")):
                return (
                    f"Scrap rate is {float(k.get('scrap_rate') or 0)*100:.1f}% "
                    f"(first-pass yield {float(k.get('fpy') or 0)*100:.1f}%). "
                    f"That is the Quality component of OEE "
                    f"({float(k.get('quality_oee') or 0)*100:.1f}%). "
                    f"{(qan.get('cards') or [{}])[0].get('message', '')}"
                )
            pareto = qan.get("defect_pareto")
            if "defect" in q or "pareto" in q:
                if isinstance(pareto, pd.DataFrame) and not pareto.empty:
                    bits = [
                        f"{r['defect_code']} ({float(r['pct'])*100:.0f}%, {float(r['rejects']):.0f} rejects)"
                        for _, r in pareto.head(3).iterrows()
                    ]
                    note = " (codes synthesized)" if qan.get("defect_synthesized") else ""
                    return "Top defects" + note + ": " + "; ".join(bits) + "."
            cpk = qan.get("cpk") or {}
            if any(w in q for w in ("cpk", "cp ", "spc", "control")):
                spc = qan.get("spc") or {}
                bits = []
                if cpk.get("ok"):
                    bits.append(
                        f"Cpk={cpk.get('cpk')} Cp={cpk.get('cp')} on {cpk.get('measurement')} "
                        f"(LSL {cpk.get('lsl')} / USL {cpk.get('usl')})."
                    )
                elif cpk.get("reason"):
                    bits.append(str(cpk["reason"]))
                if spc.get("ok"):
                    plant_spc = spc.get("plant") or {}
                    bits.append(
                        f"SPC {spc.get('metric')} I-MR: CL={plant_spc.get('cl')}, "
                        f"UCL={plant_spc.get('ucl')}, OOC={plant_spc.get('ooc_count', 0)}."
                    )
                if bits:
                    return " ".join(bits)
            cards = qan.get("cards") or []
            if cards:
                return cards[0].get("message", "Quality overlay unavailable.")
        except Exception as exc:
            return f"Quality overlay unavailable ({exc}). Load rejects / quality data."

    if money_intent:
        try:
            from modules.finance_impact import (
                asset_cost_risk_table,
                inspect_this_week_narrative,
                plant_dollar_impact,
            )

            impact = plant_dollar_impact(frame, downtime, finance_rates)
            if any(w in q for w in ("inspect", "this week", "risk")):
                matrix, _ = asset_cost_risk_table(
                    downtime if downtime is not None else frame,
                    oee_frame=frame,
                    production=frame,
                    rates=finance_rates,
                )
                if isinstance(matrix, pd.DataFrame) and not matrix.empty:
                    return inspect_this_week_narrative(matrix, finance_rates)
            if impact.get("ok"):
                return (
                    f"{impact.get('label')}: Availability loss is "
                    f"{float(impact.get('availability_hours') or 0):.1f} hours "
                    f"(${float(impact.get('availability_usd') or 0):,.0f}). "
                    f"Speed + quality hour-equivalents bring the total to "
                    f"${float(impact.get('total_usd') or 0):,.0f} at "
                    f"${float(impact.get('plant_usd_per_hour') or 0):,.0f}/hour. "
                    f"{impact.get('disclaimer')}"
                )
        except Exception as exc:
            return f"Dollar impact unavailable ({exc})."

    if maint_intent and maint_frame_ok(downtime if downtime is not None else frame):
        try:
            from modules.finance_impact import asset_cost_risk_table, inspect_this_week_narrative

            if any(w in q for w in ("inspect", "remaining risk", "this week", "pdm", "costly", "failure")):
                matrix, _ = asset_cost_risk_table(
                    downtime if downtime is not None else frame,
                    oee_frame=frame,
                    production=frame,
                    rates=finance_rates,
                )
                if isinstance(matrix, pd.DataFrame) and not matrix.empty:
                    return inspect_this_week_narrative(matrix, finance_rates)
            maint = analyze_maintenance(
                downtime if downtime is not None else frame,
                production=frame,
                oee_frame=frame,
            )
            if any(w in q for w in ("planned", "unplanned")):
                pu = maint.get("planned_vs_unplanned") or {}
                if pu.get("ok"):
                    return (
                        f"Planned downtime {float(pu.get('planned_min') or 0):.0f} min "
                        f"({float(pu.get('planned_pct') or 0)*100:.0f}%) vs unplanned "
                        f"{float(pu.get('unplanned_min') or 0):.0f} min "
                        f"({float(pu.get('unplanned_pct') or 0)*100:.0f}%) "
                        f"[{pu.get('source')} classification]."
                    )
            hours = maint.get("hours_lost") or {}
            if hours.get("narrative") and any(w in q for w in ("hour", "availability", "lost")):
                return hours["narrative"]
            cards = maint.get("cards") or []
            if cards:
                return cards[0].get("message", "")
        except Exception as exc:
            return f"Maintenance overlay unavailable ({exc})."

    if any(w in q for w in ("downtime", "pareto", "stop", "changeover", "breakdown")):
        dt = downtime if downtime is not None else frame
        try:
            pareto = downtime_pareto(dt, top_n=3)
            if not pareto.empty:
                bits = [
                    f"{r['cause']} ({float(r['pct'])*100:.0f}%, {float(r['downtime_minutes']):.0f} min)"
                    for _, r in pareto.iterrows()
                ]
                return (
                    "Top downtime drivers: " + "; ".join(bits) + ". "
                    "Attack the first 2–3 causes with SMED / root-cause / spare-parts kits."
                )
        except Exception as exc:
            return f"Downtime Pareto unavailable ({exc}). Load downtime events and retry."

    if any(w in q for w in ("mttr", "mtbf", "reliability", "repair")):
        rel = mttr_mtbf(downtime if downtime is not None else frame)
        if rel.get("ok"):
            return (
                f"MTTR ≈ {rel.get('mttr_min')} min across {rel.get('events')} events. "
                f"MTBF ≈ {rel.get('mtbf_min', 'n/a')} min. "
                "Shorten MTTR with better diagnostics; raise MTBF via planned maintenance on chronic assets."
            )
        return rel.get("reason", "Reliability metrics unavailable.")

    if any(w in q for w in ("line", "worst line", "which line")):
        by_line = summary["by_line"]
        if isinstance(by_line, pd.DataFrame) and not by_line.empty:
            worst = by_line.sort_values("oee").iloc[0]
            return (
                f"Weakest line is {worst['line_id']} at "
                f"{float(worst['oee'])*100:.1f}% OEE "
                f"(availability {float(worst['availability'])*100:.1f}%). "
                "Start with changeovers and unplanned stops on that line."
            )

    if any(w in q for w in ("machine", "bottleneck", "chronic", "asset")):
        chronic = top_chronic_machines(downtime if downtime is not None else frame, top_n=3)
        by_machine = summary["by_machine"]
        parts = []
        if isinstance(by_machine, pd.DataFrame) and not by_machine.empty:
            m = by_machine.sort_values("oee").iloc[0]
            parts.append(
                f"Lowest OEE machine: {m['machine_id']} ({float(m['oee'])*100:.1f}%)."
            )
        if not chronic.empty:
            mid = find_col(chronic, "machine_id", "machine") or chronic.columns[0]
            names = ", ".join(chronic[mid].astype(str).tolist())
            parts.append(f"Highest downtime concentration: {names}.")
        if parts:
            return " ".join(parts) + " Bundle into the next planned maintenance window."

    if any(w in q for w in ("action", "recommend", "improve", "kaizen", "plan")):
        tops = [i.get("message", "") for i in insight_list[:3]]
        return "Recommended actions:\n- " + "\n- ".join(tops)

    # Keyword match against insight messages
    tokens = set(re.findall(r"[a-z0-9]+", q))
    scored = []
    for i in insight_list:
        text = f"{i.get('title', '')} {i.get('message', '')}".lower()
        score = len(tokens & set(re.findall(r"[a-z0-9]+", text)))
        scored.append((score, i))
    scored.sort(key=lambda z: -z[0])
    if scored and scored[0][0] > 0:
        best = scored[0][1]
        return f"{best.get('title')}: {best.get('message')}"

    # Default briefing
    bullets = "\n".join(f"- {i.get('message')}" for i in insight_list[:4])
    return (
        "Here is a grounded plant briefing from current metrics:\n"
        f"{bullets}\n\n"
        "Ask about OEE, downtime Pareto, $ per lost hour, inspect this week ($ and PdM risk), scrap/FPY, or recommended actions."
    )


def _call_gemini(question: str, context: str, history: Optional[list] = None) -> str:
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")
    model_name = get_gemini_model()
    hist = ""
    for h in (history or [])[-6:]:
        hist += f"{h.get('role', 'user')}: {h.get('content', '')}\n"
    prompt = f"{SYSTEM_PROMPT}\n\nMetrics context:\n{context}\n\nChat:\n{hist}\nuser: {question}"

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=model_name, contents=prompt)
        return getattr(resp, "text", None) or str(resp)
    except Exception:
        import google.generativeai as genai_old  # type: ignore

        genai_old.configure(api_key=api_key)
        model = genai_old.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return getattr(resp, "text", None) or str(resp)


def _call_openai(question: str, context: str, history: Optional[list] = None) -> str:
    from openai import OpenAI

    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Metrics context:\n{context}"},
    ]
    for h in (history or [])[-6:]:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(
        model=get_openai_model(),
        messages=messages,
        temperature=0.2,
        max_tokens=700,
    )
    return resp.choices[0].message.content or ""


def ask_oee_question(
    question: str,
    frame: pd.DataFrame,
    downtime: pd.DataFrame | None = None,
    insights: list[dict[str, Any]] | None = None,
    history: Optional[list] = None,
    provider: str = "auto",
    quality: pd.DataFrame | None = None,
    finance_rates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer plant-manager questions with Gemini/OpenAI or strong offline fallback."""
    insight_list = insights or generate_insights(frame, downtime, quality, finance_rates)
    offline = _offline_answer(question, frame, downtime, insight_list, quality, finance_rates)
    context = build_metrics_context(frame, downtime, insight_list, quality, finance_rates)
    status = llm_status()

    provider = (provider or "auto").lower().strip()
    if provider == "auto":
        if status["gemini"]:
            provider = "gemini"
        elif status["openai"]:
            provider = "openai"
        else:
            default = get_ai_default_provider()
            if default == "gemini" and status["gemini"]:
                provider = "gemini"
            elif default == "openai" and status["openai"]:
                provider = "openai"
            else:
                provider = "offline"

    if provider == "gemini" and not status["gemini"]:
        provider = "openai" if status["openai"] else "offline"
    if provider == "openai" and not status["openai"]:
        provider = "gemini" if status["gemini"] else "offline"

    if provider == "offline":
        return {
            "ok": True,
            "source": "offline",
            "answer": offline
            + "\n\n_Offline mode. Add GEMINI_API_KEY or OPENAI_API_KEY in `.env` / Streamlit secrets for LLM answers._",
        }

    try:
        if provider == "gemini":
            answer = _call_gemini(question, context, history)
            source = "gemini"
        else:
            answer = _call_openai(question, context, history)
            source = "openai"
        # Keep a data check for metric-heavy questions
        if any(w in question.lower() for w in ("oee", "mttr", "downtime", "%", "line", "scrap", "inspect")):
            answer = f"{answer}\n\n---\nData check:\n{offline}"
        return {"ok": True, "source": source, "answer": answer}
    except Exception as exc:
        return {
            "ok": True,
            "source": "offline",
            "answer": f"{offline}\n\n_({provider} error: {exc})_",
        }
