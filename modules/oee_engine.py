"""OEE calculation engine: Availability, Performance, Quality, OEE."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from modules.quality_checks import find_col


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def compute_oee_row(
    planned_time_min: float,
    downtime_min: float,
    ideal_rate_per_min: float,
    total_count: float,
    good_count: float,
) -> dict[str, float]:
    """Classic OEE from shift-level aggregates."""
    planned = max(float(planned_time_min or 0), 0.0)
    down = max(float(downtime_min or 0), 0.0)
    run_time = max(planned - down, 0.0)
    ideal = max(float(ideal_rate_per_min or 0), 0.0)
    total = max(float(total_count or 0), 0.0)
    good = max(float(good_count or 0), 0.0)
    if good > total:
        good = total

    availability = (run_time / planned) if planned > 0 else 0.0
    performance = (total / (run_time * ideal)) if run_time > 0 and ideal > 0 else 0.0
    performance = min(performance, 1.5)  # cap runaway sensor errors
    quality = (good / total) if total > 0 else 0.0
    oee = availability * performance * quality
    return {
        "run_time_min": round(run_time, 2),
        "availability": round(availability, 4),
        "performance": round(performance, 4),
        "quality": round(quality, 4),
        "oee": round(oee, 4),
    }


def prepare_oee_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and compute OEE metrics per row."""
    out = df.copy()

    planned = find_col(out, "planned_time_min", "planned_minutes", "available_time_min", "shift_minutes")
    downtime = find_col(out, "downtime_minutes", "downtime_min", "unplanned_downtime_min")
    ideal = find_col(out, "ideal_rate", "ideal_rate_per_min", "design_rate", "target_rate")
    total = find_col(out, "total_count", "total_units", "produced_count", "output_count")
    good = find_col(out, "good_count", "good_units", "good_output")
    rejects = find_col(out, "reject_count", "rejects", "scrap_count")

    # Aggregate downtime if event-level rows collapsed badly
    if downtime is None and "downtime_minutes" in out.columns:
        downtime = "downtime_minutes"

    if planned is None:
        out["planned_time_min"] = 480.0
        planned = "planned_time_min"
    if downtime is None:
        out["downtime_minutes"] = 0.0
        downtime = "downtime_minutes"
    if ideal is None:
        out["ideal_rate"] = 10.0
        ideal = "ideal_rate"
    if total is None:
        if good is not None and rejects is not None:
            out["total_count"] = pd.to_numeric(out[good], errors="coerce").fillna(0) + pd.to_numeric(
                out[rejects], errors="coerce"
            ).fillna(0)
            total = "total_count"
        else:
            out["total_count"] = 0.0
            total = "total_count"
    if good is None:
        if rejects is not None:
            out["good_count"] = (
                pd.to_numeric(out[total], errors="coerce").fillna(0)
                - pd.to_numeric(out[rejects], errors="coerce").fillna(0)
            ).clip(lower=0)
            good = "good_count"
        else:
            out["good_count"] = pd.to_numeric(out[total], errors="coerce").fillna(0)
            good = "good_count"

    for c in (planned, downtime, ideal, total, good):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    metrics = out.apply(
        lambda r: compute_oee_row(r[planned], r[downtime], r[ideal], r[total], r[good]),
        axis=1,
        result_type="expand",
    )
    for c in metrics.columns:
        out[c] = metrics[c]
    out["loss_availability"] = (1 - out["availability"]).clip(lower=0)
    out["loss_performance"] = (out["availability"] * (1 - out["performance"])).clip(lower=0)
    out["loss_quality"] = (
        out["availability"] * out["performance"] * (1 - out["quality"])
    ).clip(lower=0)
    return out


def aggregate_oee(
    df: pd.DataFrame,
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Aggregate OEE by shift / line / machine using weighted components."""
    work = prepare_oee_frame(df) if "oee" not in df.columns else df.copy()
    planned = find_col(work, "planned_time_min", "planned_minutes") or "planned_time_min"
    downtime = find_col(work, "downtime_minutes", "downtime_min") or "downtime_minutes"
    ideal = find_col(work, "ideal_rate", "ideal_rate_per_min") or "ideal_rate"
    total = find_col(work, "total_count", "total_units") or "total_count"
    good = find_col(work, "good_count", "good_units") or "good_count"

    if not group_cols:
        group_cols = [c for c in ["line_id", "machine_id", "shift", "shift_date"] if c in work.columns]
    if not group_cols:
        # Plant-level single row
        row = compute_oee_row(
            work[planned].sum(),
            work[downtime].sum(),
            float(work[ideal].mean() or 0),
            work[total].sum(),
            work[good].sum(),
        )
        return pd.DataFrame([{**row, "scope": "plant"}])

    rows = []
    for keys, g in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        rec.update(
            compute_oee_row(
                g[planned].sum(),
                g[downtime].sum(),
                float(g[ideal].mean() or 0),
                g[total].sum(),
                g[good].sum(),
            )
        )
        rec["shifts"] = len(g)
        rec["total_count"] = float(g[total].sum())
        rec["good_count"] = float(g[good].sum())
        rec["downtime_minutes"] = float(g[downtime].sum())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("oee")


def oee_summary(df: pd.DataFrame) -> dict[str, Any]:
    work = prepare_oee_frame(df)
    plant = aggregate_oee(work, group_cols=None).iloc[0].to_dict()
    by_line = aggregate_oee(work, [c for c in ["line_id"] if c in work.columns]) if "line_id" in work.columns else pd.DataFrame()
    by_machine = (
        aggregate_oee(work, [c for c in ["machine_id"] if c in work.columns])
        if "machine_id" in work.columns
        else pd.DataFrame()
    )
    by_shift = (
        aggregate_oee(work, [c for c in ["shift"] if c in work.columns])
        if "shift" in work.columns
        else pd.DataFrame()
    )
    return {
        "plant": plant,
        "by_line": by_line,
        "by_machine": by_machine,
        "by_shift": by_shift,
        "frame": work,
        "world_class_gap": round(max(0.85 - float(plant.get("oee", 0)), 0), 4),
    }
