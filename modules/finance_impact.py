"""Management-estimate $ impact of lost OEE hours (Availability + optional P/Q)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from modules.downtime_analysis import downtime_pareto
from modules.maintenance_analysis import (
    asset_risk_table,
    availability_loss_hours,
    remaining_risk_ranking,
)
from modules.oee_engine import oee_summary, prepare_oee_frame
from modules.quality_checks import find_col

# Synthetic demo defaults so sample plant data shows $ without extra files.
DEFAULT_PLANT_USD_PER_HOUR = 850.0
DEFAULT_LINE_USD_PER_HOUR: dict[str, float] = {
    "L1": 850.0,
    "L2": 1100.0,
    "L3": 720.0,
}
ESTIMATE_LABEL = "Management estimate"
FINANCE_LABEL = "Finance-aligned rates"


def default_finance_rates() -> dict[str, Any]:
    return {
        "plant_usd_per_hour": DEFAULT_PLANT_USD_PER_HOUR,
        "usd_per_good_unit": 0.0,
        "line_usd_per_hour": dict(DEFAULT_LINE_USD_PER_HOUR),
        "rates_match_finance": False,
        "include_performance_quality_hours": True,
    }


def normalize_rates(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    base = default_finance_rates()
    if not isinstance(raw, dict):
        return base
    plant = pd.to_numeric(raw.get("plant_usd_per_hour"), errors="coerce")
    if pd.notna(plant) and float(plant) >= 0:
        base["plant_usd_per_hour"] = float(plant)
    unit = pd.to_numeric(raw.get("usd_per_good_unit"), errors="coerce")
    if pd.notna(unit) and float(unit) >= 0:
        base["usd_per_good_unit"] = float(unit)
    lines = raw.get("line_usd_per_hour") or {}
    if isinstance(lines, dict):
        cleaned: dict[str, float] = {}
        for k, v in lines.items():
            n = pd.to_numeric(v, errors="coerce")
            if pd.notna(n) and float(n) >= 0:
                cleaned[str(k)] = float(n)
        if cleaned:
            base["line_usd_per_hour"] = cleaned
    base["rates_match_finance"] = bool(raw.get("rates_match_finance"))
    if "include_performance_quality_hours" in raw:
        base["include_performance_quality_hours"] = bool(raw.get("include_performance_quality_hours"))
    return base


def rate_disclaimer(rates: Optional[dict[str, Any]] = None) -> str:
    r = normalize_rates(rates)
    if r["rates_match_finance"]:
        return (
            f"{FINANCE_LABEL}: dollar figures use the $/hour (and optional $/good unit) "
            "you entered as matching finance."
        )
    return (
        f"{ESTIMATE_LABEL}: dollar figures use plant/line $/hour you entered "
        "(sample defaults if unchanged). They match the P&L only when rates come from finance."
    )


def _line_rate(line: Any, rates: dict[str, Any]) -> float:
    table = rates.get("line_usd_per_hour") or {}
    key = None if line is None or (isinstance(line, float) and np.isnan(line)) else str(line)
    if key and key in table:
        return float(table[key])
    return float(rates.get("plant_usd_per_hour") or 0.0)


def _hours_from_minutes(minutes: float) -> float:
    return max(float(minutes or 0.0), 0.0) / 60.0


def lost_hours_from_oee(frame: pd.DataFrame) -> dict[str, Any]:
    """Convert OEE loss components into hour-equivalents of planned time.

    Availability hours = (1-A) * planned_hours
    Performance hours  = A*(1-P) * planned_hours   (speed loss as hours)
    Quality hours      = A*P*(1-Q) * planned_hours (yield loss as hours)
    """
    empty = {
        "ok": False,
        "planned_hours": 0.0,
        "availability_hours": 0.0,
        "performance_hours": 0.0,
        "quality_hours": 0.0,
        "oee_hours": 0.0,
        "availability": None,
        "performance": None,
        "quality": None,
        "oee": None,
    }
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        empty["reason"] = "No production frame."
        return empty
    work = prepare_oee_frame(frame)
    planned_col = find_col(work, "planned_time_min", "planned_minutes") or "planned_time_min"
    planned_hours = float(pd.to_numeric(work[planned_col], errors="coerce").fillna(0).sum()) / 60.0
    summary = oee_summary(work)
    plant = summary["plant"]
    avail = float(plant.get("availability") or 0)
    perf = float(plant.get("performance") or 0)
    qual = float(plant.get("quality") or 0)
    oee = float(plant.get("oee") or 0)
    return {
        "ok": True,
        "planned_hours": round(planned_hours, 2),
        "availability_hours": round((1.0 - avail) * planned_hours, 2),
        "performance_hours": round(avail * (1.0 - min(perf, 1.0)) * planned_hours, 2),
        "quality_hours": round(avail * min(perf, 1.0) * (1.0 - min(qual, 1.0)) * planned_hours, 2),
        "oee_hours": round((1.0 - min(oee, 1.0)) * planned_hours, 2),
        "availability": round(avail, 4),
        "performance": round(perf, 4),
        "quality": round(qual, 4),
        "oee": round(oee, 4),
        "frame": work,
        "summary": summary,
    }


def _usd(hours: float, rate: float) -> float:
    return round(max(hours, 0.0) * max(rate, 0.0), 2)


def plant_dollar_impact(
    frame: Optional[pd.DataFrame],
    downtime: Optional[pd.DataFrame] = None,
    rates: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    r = normalize_rates(rates)
    plant_rate = float(r["plant_usd_per_hour"])
    hours = lost_hours_from_oee(frame) if frame is not None and isinstance(frame, pd.DataFrame) else {
        "ok": False,
        "planned_hours": 0.0,
        "availability_hours": 0.0,
        "performance_hours": 0.0,
        "quality_hours": 0.0,
        "oee_hours": 0.0,
    }
    include_pq = bool(r.get("include_performance_quality_hours"))
    avail_h = float(hours.get("availability_hours") or 0)
    perf_h = float(hours.get("performance_hours") or 0) if include_pq else 0.0
    qual_h = float(hours.get("quality_hours") or 0) if include_pq else 0.0

    unit = float(r.get("usd_per_good_unit") or 0)
    quality_from_units = 0.0
    if unit > 0 and frame is not None and isinstance(frame, pd.DataFrame) and not frame.empty:
        work = hours.get("frame")
        if work is None or not isinstance(work, pd.DataFrame):
            work = prepare_oee_frame(frame)
        rej = find_col(work, "reject_count", "rejects", "scrap_count")
        if rej:
            quality_from_units = float(pd.to_numeric(work[rej], errors="coerce").fillna(0).sum()) * unit

    avail_usd = _usd(avail_h, plant_rate)
    perf_usd = _usd(perf_h, plant_rate)
    qual_usd = round(quality_from_units, 2) if unit > 0 else _usd(qual_h, plant_rate)
    total_usd = round(avail_usd + perf_usd + qual_usd, 2)

    dt_hours = availability_loss_hours(downtime, frame) if downtime is not None else {}

    return {
        "ok": bool(hours.get("ok")),
        "rates": r,
        "disclaimer": rate_disclaimer(r),
        "label": FINANCE_LABEL if r["rates_match_finance"] else ESTIMATE_LABEL,
        "plant_usd_per_hour": plant_rate,
        "usd_per_good_unit": unit,
        "hours": hours,
        "availability_hours": avail_h,
        "performance_hours": perf_h,
        "quality_hours": qual_h if unit <= 0 else None,
        "availability_usd": avail_usd,
        "performance_usd": perf_usd,
        "quality_usd": qual_usd,
        "total_usd": total_usd,
        "downtime_hours_meta": dt_hours,
    }


def pareto_with_dollars(
    downtime: Optional[pd.DataFrame],
    rates: Optional[dict[str, Any]] = None,
    production: Optional[pd.DataFrame] = None,
    top_n: int = 12,
) -> pd.DataFrame:
    """Downtime Pareto with $ using line rate when line_id is present, else plant default."""
    if downtime is None or not isinstance(downtime, pd.DataFrame) or downtime.empty:
        return pd.DataFrame()
    r = normalize_rates(rates)
    try:
        pareto = downtime_pareto(downtime, top_n=top_n)
    except Exception:
        return pd.DataFrame()
    if pareto is None or not isinstance(pareto, pd.DataFrame) or pareto.empty:
        return pd.DataFrame()

    minutes = find_col(downtime, "downtime_minutes", "downtime_min", "duration_min")
    cause = find_col(downtime, "downtime_code", "reason_code", "failure_code", "downtime_category", "category")
    line = find_col(downtime, "line_id", "line")
    if minutes is None:
        pareto["usd_lost"] = _hours_from_minutes(pareto["downtime_minutes"]) * float(r["plant_usd_per_hour"])
        return pareto

    work = downtime.copy()
    work[minutes] = pd.to_numeric(work[minutes], errors="coerce").fillna(0)
    if cause is None:
        work["_cause"] = "UNSPECIFIED"
        cause = "_cause"
    if line:
        work["_rate"] = work[line].map(lambda v: _line_rate(v, r))
        work["_usd"] = work[minutes] / 60.0 * work["_rate"]
        g = (
            work.groupby(cause, dropna=False)
            .agg(
                events=(minutes, "count"),
                downtime_minutes=(minutes, "sum"),
                usd_lost=("_usd", "sum"),
            )
            .reset_index()
            .rename(columns={cause: "cause"})
        )
        g = g.sort_values("downtime_minutes", ascending=False).head(top_n)
        total = float(g["downtime_minutes"].sum()) or 1.0
        g["pct"] = g["downtime_minutes"] / total
        g["cum_pct"] = g["pct"].cumsum()
        g["usd_lost"] = g["usd_lost"].round(2)
        return g

    pareto = pareto.copy()
    pareto["usd_lost"] = (pareto["downtime_minutes"] / 60.0 * float(r["plant_usd_per_hour"])).round(2)
    return pareto


def _availability_hours_by_asset(frame: Optional[pd.DataFrame], downtime: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame is not None and isinstance(frame, pd.DataFrame) and not frame.empty:
        work = prepare_oee_frame(frame)
        machine = find_col(work, "machine_id", "machine", "asset_id")
        line = find_col(work, "line_id", "line")
        planned = find_col(work, "planned_time_min", "planned_minutes") or "planned_time_min"
        if machine:
            keys = [machine] + ([line] if line and line != machine else [])
            for key, g in work.groupby(keys, dropna=False):
                if not isinstance(key, tuple):
                    key = (key,)
                rec = dict(zip(keys, key))
                ph = float(pd.to_numeric(g[planned], errors="coerce").fillna(0).sum()) / 60.0
                avail = float(g["availability"].mean()) if "availability" in g.columns else 0.0
                rows.append(
                    {
                        "machine_id": rec.get(machine),
                        "line_id": rec.get(line) if line else None,
                        "planned_hours": round(ph, 2),
                        "availability": round(avail, 4),
                        "availability_hours_lost": round((1.0 - avail) * ph, 2),
                    }
                )
            if rows:
                return pd.DataFrame(rows)

    if downtime is not None and isinstance(downtime, pd.DataFrame) and not downtime.empty:
        minutes = find_col(downtime, "downtime_minutes", "downtime_min", "duration_min")
        machine = find_col(downtime, "machine_id", "machine", "asset_id")
        line = find_col(downtime, "line_id", "line")
        if minutes and machine:
            work = downtime.copy()
            work[minutes] = pd.to_numeric(work[minutes], errors="coerce").fillna(0)
            keys = [machine] + ([line] if line and line != machine else [])
            g = work.groupby(keys, dropna=False)[minutes].sum().reset_index()
            g["machine_id"] = g[machine]
            g["line_id"] = g[line] if line else None
            g["availability_hours_lost"] = (g[minutes] / 60.0).round(2)
            g["planned_hours"] = np.nan
            g["availability"] = np.nan
            return g[["machine_id", "line_id", "planned_hours", "availability", "availability_hours_lost"]]
    return pd.DataFrame()


def asset_cost_risk_table(
    downtime: Optional[pd.DataFrame],
    oee_frame: Optional[pd.DataFrame] = None,
    production: Optional[pd.DataFrame] = None,
    rates: Optional[dict[str, Any]] = None,
    top_n: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Side-by-side Availability hours lost | $ lost | PdM/failure risk."""
    r = normalize_rates(rates)
    hours_df = _availability_hours_by_asset(oee_frame, downtime)
    iso_meta: dict[str, Any] = {"ok": False}
    risk = pd.DataFrame()
    if downtime is not None and isinstance(downtime, pd.DataFrame) and not downtime.empty:
        risk, iso_meta = remaining_risk_ranking(downtime, production=production, top_n=max(top_n, 40))
        if risk is None or not isinstance(risk, pd.DataFrame):
            risk = pd.DataFrame()
        if risk.empty:
            risk = asset_risk_table(downtime)

    if hours_df is None or not isinstance(hours_df, pd.DataFrame) or hours_df.empty:
        if risk is None or not isinstance(risk, pd.DataFrame) or risk.empty:
            return pd.DataFrame(), iso_meta
        out = risk.copy()
        out["availability_hours_lost"] = (pd.to_numeric(out.get("downtime_minutes"), errors="coerce").fillna(0) / 60.0).round(2)
    else:
        out = hours_df.copy()
        if isinstance(risk, pd.DataFrame) and not risk.empty:
            keep = [
                c
                for c in (
                    "machine_id",
                    "remaining_risk",
                    "rule_score",
                    "iso_anomaly_rate_pct",
                    "iso_boost",
                    "mttr_min",
                    "failure_freq_per_day",
                    "unplanned_share",
                    "why",
                    "action",
                    "events",
                    "downtime_minutes",
                )
                if c in risk.columns
            ]
            out = out.merge(risk[keep], on="machine_id", how="left")

    if "remaining_risk" not in out.columns:
        out["remaining_risk"] = 0.0
    out["remaining_risk"] = pd.to_numeric(out["remaining_risk"], errors="coerce").fillna(0)
    out["availability_hours_lost"] = pd.to_numeric(out.get("availability_hours_lost"), errors="coerce").fillna(0)
    out["usd_per_hour"] = out.apply(lambda row: _line_rate(row.get("line_id"), r), axis=1)
    out["usd_lost"] = (out["availability_hours_lost"] * out["usd_per_hour"]).round(2)

    def _norm(s: pd.Series) -> pd.Series:
        v = pd.to_numeric(s, errors="coerce").fillna(0.0)
        span = float(v.max() - v.min())
        if span <= 1e-12:
            return pd.Series(np.zeros(len(v)), index=v.index)
        return (v - float(v.min())) / span

    out["priority_score"] = (55.0 * _norm(out["usd_lost"]) + 45.0 * _norm(out["remaining_risk"])).round(1)
    out["label"] = FINANCE_LABEL if r["rates_match_finance"] else ESTIMATE_LABEL
    out = out.sort_values("priority_score", ascending=False).head(top_n)
    return out.reset_index(drop=True), iso_meta


def inspect_this_week_narrative(matrix: pd.DataFrame, rates: Optional[dict[str, Any]] = None) -> str:
    if matrix is None or not isinstance(matrix, pd.DataFrame) or matrix.empty:
        return "No asset ranking yet — load downtime events (or sample plant data)."
    r = normalize_rates(rates)
    top = matrix.iloc[0]
    mid = top.get("machine_id")
    line = top.get("line_id")
    hours = float(top.get("availability_hours_lost") or 0)
    usd = float(top.get("usd_lost") or 0)
    risk = float(top.get("remaining_risk") or 0)
    why = top.get("why") or ""
    action = top.get("action") or "Inspect this week."
    label = FINANCE_LABEL if r["rates_match_finance"] else ESTIMATE_LABEL
    bits = [
        f"Inspect this week: {mid} on {line} is both costly and high failure-risk.",
        f"Availability hours lost {hours:.1f} h · ${usd:,.0f} ({label}) · PdM risk {risk:.0f}/100.",
    ]
    if why:
        bits.append(str(why) + ".")
    bits.append(str(action))
    extra = []
    for _, row in matrix.head(3).iterrows():
        extra.append(
            f"{row.get('machine_id')} (${float(row.get('usd_lost') or 0):,.0f}, "
            f"risk {float(row.get('remaining_risk') or 0):.0f})"
        )
    bits.append("Top 3 by $ × risk: " + "; ".join(extra) + ".")
    return " ".join(bits)
