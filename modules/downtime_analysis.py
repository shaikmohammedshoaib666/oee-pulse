"""Downtime Pareto analysis and MTTR / MTBF lite metrics."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from modules.quality_checks import find_col


def downtime_pareto(df: pd.DataFrame, code_col: Optional[str] = None, top_n: int = 12) -> pd.DataFrame:
    """Pareto of downtime minutes by code / category."""
    minutes = find_col(df, "downtime_minutes", "downtime_min", "duration_min")
    if minutes is None:
        raise ValueError("No downtime_minutes column found")
    code = code_col or find_col(
        df, "downtime_code", "reason_code", "failure_code", "downtime_category", "category"
    )
    if code is None:
        code = "downtime_code"
        work = df.copy()
        work[code] = "UNSPECIFIED"
    else:
        work = df.copy()

    work[minutes] = pd.to_numeric(work[minutes], errors="coerce").fillna(0)
    g = (
        work.groupby(code, dropna=False)[minutes]
        .agg(events="count", downtime_minutes="sum")
        .reset_index()
        .sort_values("downtime_minutes", ascending=False)
    )
    total = g["downtime_minutes"].sum() or 1.0
    g["pct"] = g["downtime_minutes"] / total
    g["cum_pct"] = g["pct"].cumsum()
    g = g.head(top_n)
    return g.rename(columns={code: "cause"})


def mttr_mtbf(df: pd.DataFrame) -> dict[str, Any]:
    """
    Lite MTTR / MTBF from downtime events.
    MTTR = mean downtime minutes per event
    MTBF ≈ mean operating time between failures (if timestamps / planned time available)
    """
    minutes = find_col(df, "downtime_minutes", "downtime_min", "duration_min")
    if minutes is None or len(df) == 0:
        return {"ok": False, "reason": "no downtime events"}

    d = pd.to_numeric(df[minutes], errors="coerce").dropna()
    d = d[d > 0]
    if len(d) == 0:
        return {"ok": False, "reason": "no positive downtime"}

    mttr = float(d.mean())
    out: dict[str, Any] = {
        "ok": True,
        "events": int(len(d)),
        "mttr_min": round(mttr, 2),
        "total_downtime_min": round(float(d.sum()), 2),
    }

    start = find_col(df, "start_time", "timestamp", "event_start")
    end = find_col(df, "end_time", "event_end")
    machine = find_col(df, "machine_id", "machine", "asset_id")

    if start and machine:
        work = df.copy()
        work[start] = pd.to_datetime(work[start], errors="coerce")
        work = work.dropna(subset=[start, machine]).sort_values([machine, start])
        gaps = []
        for _, g in work.groupby(machine):
            ts = g[start].diff().dt.total_seconds() / 60.0
            gaps.extend(ts.dropna().tolist())
        if gaps:
            # Approximate MTBF as mean gap between failure starts minus MTTR
            mean_gap = float(np.mean(gaps))
            mtbf = max(mean_gap - mttr, 0)
            out["mtbf_min"] = round(mtbf, 2)
            out["mean_interarrival_min"] = round(mean_gap, 2)

    planned = find_col(df, "planned_time_min", "planned_minutes")
    if "mtbf_min" not in out and planned is not None:
        # Shift-level approximation: operating time / failure count
        p = pd.to_numeric(df[planned], errors="coerce").fillna(0).sum()
        run = max(p - float(d.sum()), 0)
        out["mtbf_min"] = round(run / max(len(d), 1), 2)
        out["mtbf_method"] = "planned_time_proxy"

    if start and end:
        work = df.copy()
        work[start] = pd.to_datetime(work[start], errors="coerce")
        work[end] = pd.to_datetime(work[end], errors="coerce")
        dur = (work[end] - work[start]).dt.total_seconds() / 60.0
        dur = dur.dropna()
        dur = dur[dur > 0]
        if len(dur):
            out["mttr_from_timestamps_min"] = round(float(dur.mean()), 2)

    return out


def downtime_by_dimension(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    minutes = find_col(df, "downtime_minutes", "downtime_min")
    if minutes is None or dim not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby(dim)[minutes]
        .agg(events="count", downtime_minutes="sum")
        .reset_index()
        .sort_values("downtime_minutes", ascending=False)
    )
    return g


def top_chronic_machines(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    machine = find_col(df, "machine_id", "machine")
    minutes = find_col(df, "downtime_minutes", "downtime_min")
    if not machine or not minutes:
        return pd.DataFrame()
    g = (
        df.groupby(machine)[minutes]
        .agg(events="count", downtime_minutes="sum", mean_min="mean")
        .reset_index()
        .sort_values("downtime_minutes", ascending=False)
        .head(top_n)
    )
    return g
