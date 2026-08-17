"""Manager-language business insights for OEE and downtime."""

from __future__ import annotations

from typing import Any

import pandas as pd

from modules.downtime_analysis import downtime_pareto, mttr_mtbf, top_chronic_machines
from modules.oee_engine import oee_summary
from modules.quality_checks import find_col


def generate_insights(
    production_oee: pd.DataFrame,
    downtime: pd.DataFrame | None = None,
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

    # Biggest loss pillar
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
                        f"Cumulative top causes reach {float(pareto['cum_pct'].iloc[min(2, len(pareto)-1)])*100:.0f}% "
                        f"by the 3rd cause — classic Pareto opportunity."
                    ),
                }
            )
            # Changeover specific language if present
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
                avail_proxy = reliability["mtbf_min"] / (reliability["mtbf_min"] + reliability["mttr_min"])
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

    # Production planning hint
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

    # Sort high → low
    order = {"high": 0, "medium": 1, "low": 2}
    insights.sort(key=lambda x: order.get(x["priority"], 9))
    return insights
