"""Maintenance department overlay: reliability, risk ranking, planned vs unplanned."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from modules.downtime_analysis import downtime_pareto, mttr_mtbf
from modules.oee_engine import oee_summary
from modules.quality_checks import find_col

SENSOR_TOKENS = (
    "vibration",
    "temp",
    "temperature",
    "current",
    "pressure",
    "rms",
    "sensor",
    "amp",
    "motor",
)
SENSOR_EXCLUDE = ("count", "minute", "rate", "oee", "scrap", "reject", "planned", "good", "total")
PLANNED_TOKENS = (
    "planned",
    "pm07",
    "preventive",
    "changeover",
    "chg01",
    "setup",
    "smed",
    "scheduled",
    "pm ",
)
UNPLANNED_TOKENS = (
    "breakdown",
    "brk",
    "failure",
    "unplanned",
    "mechanical",
    "electrical",
    "jam",
    "fault",
)


def frame_ok(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty


def _first_frame(*frames: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    for f in frames:
        if frame_ok(f):
            return f
    return None


def _norm01(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").fillna(0.0)
    span = float(v.max() - v.min())
    if span <= 1e-12:
        return pd.Series(np.zeros(len(v)), index=v.index)
    return (v - float(v.min())) / span


def _window(df: pd.DataFrame, date_col: Optional[str], days: int = 7) -> pd.DataFrame:
    if not date_col or date_col not in df.columns:
        return df
    d = pd.to_datetime(df[date_col], errors="coerce")
    mx = d.max()
    if pd.isna(mx):
        return df
    mask = d >= (mx - pd.Timedelta(days=days))
    out = df.loc[mask.fillna(False)]
    return out if not out.empty else df


def classify_planned_unplanned(df: pd.DataFrame) -> pd.DataFrame:
    """Add event_class Planned/Unplanned from a column or changeover-vs-breakdown heuristic."""
    work = df.copy()
    if work.empty:
        work["event_class"] = pd.Series(dtype=str)
        work["is_planned"] = pd.Series(dtype=bool)
        return work

    flag = find_col(work, "is_planned", "planned_flag", "planned")
    etype = find_col(work, "event_class", "event_type", "downtime_type", "stop_type", "failure_type")
    text_cols = [
        c
        for c in (
            find_col(work, "downtime_category", "category", "reason"),
            find_col(work, "downtime_code", "reason_code", "failure_code"),
            etype,
        )
        if c
    ]
    blob = pd.Series([""] * len(work), index=work.index)
    for c in text_cols:
        blob = blob + " " + work[c].astype(str).str.lower()

    planned = pd.Series(False, index=work.index)
    if flag:
        raw = work[flag]
        if raw.dtype == bool:
            planned = raw.fillna(False)
        else:
            planned = (
                raw.astype(str)
                .str.strip()
                .str.lower()
                .isin({"1", "true", "yes", "y", "planned", "p"})
            )
    elif etype:
        planned = work[etype].astype(str).str.lower().str.contains(
            "plan|changeover|setup|pm|prevent", regex=True, na=False
        )
    else:
        planned = blob.apply(lambda s: any(t in s for t in PLANNED_TOKENS))

    work["is_planned"] = planned.astype(bool)
    work["event_class"] = np.where(work["is_planned"], "Planned", "Unplanned")
    work["_class_source"] = "column" if (flag or etype) else "heuristic"
    return work


def _sensor_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.select_dtypes(include=[np.number]).columns:
        cl = str(c).lower()
        if any(x in cl for x in SENSOR_EXCLUDE):
            continue
        if any(t in cl for t in SENSOR_TOKENS):
            cols.append(c)
    return cols


def planned_vs_unplanned(df: pd.DataFrame) -> dict[str, Any]:
    if not frame_ok(df):
        return {"ok": False, "reason": "Downtime events table missing.", "by_line": pd.DataFrame()}
    minutes = find_col(df, "downtime_minutes", "downtime_min", "duration_min")
    if minutes is None:
        return {"ok": False, "reason": "No downtime_minutes column.", "by_line": pd.DataFrame()}
    work = classify_planned_unplanned(df)
    work[minutes] = pd.to_numeric(work[minutes], errors="coerce").fillna(0)
    total = float(work[minutes].sum())
    pmin = float(work.loc[work["is_planned"], minutes].sum())
    umin = float(work.loc[~work["is_planned"], minutes].sum())
    line = find_col(work, "line_id", "line")
    by_line = pd.DataFrame()
    if line:
        by_line = (
            work.groupby([line, "event_class"], dropna=False)[minutes]
            .sum()
            .reset_index()
            .rename(columns={line: "line_id", minutes: "downtime_minutes"})
        )
    return {
        "ok": True,
        "source": str(work["_class_source"].iloc[0]) if len(work) else "heuristic",
        "total_min": round(total, 1),
        "planned_min": round(pmin, 1),
        "unplanned_min": round(umin, 1),
        "planned_pct": round(pmin / total, 4) if total else 0.0,
        "unplanned_pct": round(umin / total, 4) if total else 0.0,
        "planned_events": int(work["is_planned"].sum()),
        "unplanned_events": int((~work["is_planned"]).sum()),
        "by_line": by_line,
        "classified": work,
    }


def reason_pareto_by(
    df: pd.DataFrame,
    by: str = "machine_id",
    top_n: int = 8,
) -> pd.DataFrame:
    """Pareto of downtime reasons within each machine or line."""
    if not frame_ok(df):
        return pd.DataFrame()
    minutes = find_col(df, "downtime_minutes", "downtime_min", "duration_min")
    dim = find_col(df, by, "machine", "asset_id", "line")
    code = find_col(
        df, "downtime_code", "downtime_category", "reason_code", "failure_code", "category"
    )
    if minutes is None or dim is None:
        return pd.DataFrame()
    if code is None:
        work = df.copy()
        work["cause"] = "UNSPECIFIED"
        code = "cause"
    else:
        work = df.copy()
    work[minutes] = pd.to_numeric(work[minutes], errors="coerce").fillna(0)
    g = (
        work.groupby([dim, code], dropna=False)[minutes]
        .agg(events="count", downtime_minutes="sum")
        .reset_index()
        .rename(columns={dim: "asset", code: "cause"})
    )
    if g.empty:
        return g
    g["pct_within_asset"] = g.groupby("asset")["downtime_minutes"].transform(
        lambda s: s / (s.sum() or 1.0)
    )
    g = g.sort_values(["asset", "downtime_minutes"], ascending=[True, False])
    g = g.groupby("asset", group_keys=False).head(top_n)
    g["scope"] = by
    return g


def longest_events(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if not frame_ok(df):
        return pd.DataFrame()
    minutes = find_col(df, "downtime_minutes", "downtime_min", "duration_min")
    if minutes is None:
        return pd.DataFrame()
    work = classify_planned_unplanned(df)
    work[minutes] = pd.to_numeric(work[minutes], errors="coerce").fillna(0)
    cols = [
        c
        for c in (
            find_col(work, "event_id"),
            find_col(work, "shift_date", "date"),
            find_col(work, "line_id", "line"),
            find_col(work, "machine_id", "machine", "asset_id"),
            find_col(work, "downtime_code"),
            find_col(work, "downtime_category", "category"),
            minutes,
            "event_class",
            find_col(work, "start_time"),
            find_col(work, "end_time"),
        )
        if c
    ]
    out = work.loc[work[minutes] > 0, cols].sort_values(minutes, ascending=False).head(top_n)
    return out.rename(columns={minutes: "downtime_minutes"})


def asset_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-asset MTTR, MTBF, failure frequency, longest event, unplanned share."""
    if not frame_ok(df):
        return pd.DataFrame()
    minutes = find_col(df, "downtime_minutes", "downtime_min", "duration_min")
    machine = find_col(df, "machine_id", "machine", "asset_id")
    if minutes is None or machine is None:
        return pd.DataFrame()

    work = classify_planned_unplanned(df)
    work[minutes] = pd.to_numeric(work[minutes], errors="coerce").fillna(0)
    line = find_col(work, "line_id", "line")
    date_col = find_col(work, "shift_date", "start_time", "date")
    start = find_col(work, "start_time", "timestamp", "event_start")

    keys = [machine] + ([line] if line and line != machine else [])
    rows: list[dict[str, Any]] = []
    for key, g in work.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(keys, key))
        d = g[minutes]
        pos = d[d > 0]
        events = int(len(pos))
        total = float(d.sum())
        mttr = float(pos.mean()) if events else 0.0
        longest = float(pos.max()) if events else 0.0
        unpl = float(g.loc[~g["is_planned"], minutes].sum())
        days = 1.0
        if date_col:
            ts = pd.to_datetime(g[date_col], errors="coerce").dropna()
            if len(ts):
                span = (ts.max() - ts.min()).total_seconds() / 86400.0
                days = max(span, 1.0)
        freq = events / days
        mtbf = np.nan
        if start and events > 1:
            gs = g.copy()
            gs[start] = pd.to_datetime(gs[start], errors="coerce")
            gs = gs.dropna(subset=[start]).sort_values(start)
            gaps = gs[start].diff().dt.total_seconds() / 60.0
            gaps = gaps.dropna()
            if len(gaps):
                mtbf = max(float(gaps.mean()) - mttr, 0.0)
        rec.update(
            {
                "machine_id": rec.get(machine),
                "line_id": rec.get(line) if line else None,
                "events": events,
                "downtime_minutes": round(total, 1),
                "mttr_min": round(mttr, 2),
                "mtbf_min": None if np.isnan(mtbf) else round(mtbf, 2),
                "failure_freq_per_day": round(freq, 3),
                "longest_event_min": round(longest, 1),
                "unplanned_minutes": round(unpl, 1),
                "unplanned_share": round(unpl / total, 4) if total else 0.0,
            }
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rule_score"] = (
        35.0 * _norm01(out["failure_freq_per_day"])
        + 25.0 * _norm01(out["mttr_min"])
        + 20.0 * _norm01(out["longest_event_min"])
        + 20.0 * out["unplanned_share"].clip(0, 1)
    )
    return out.sort_values("rule_score", ascending=False)


def sensor_anomaly_by_asset(production: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Optional IsolationForest on numeric sensor columns, rolled up by machine."""
    meta: dict[str, Any] = {"ok": False, "features": []}
    if not frame_ok(production):
        meta["reason"] = "no production/sensor frame"
        return pd.DataFrame(), meta
    feats = _sensor_columns(production)
    machine = find_col(production, "machine_id", "machine", "asset_id")
    if len(feats) < 2 or machine is None:
        meta["reason"] = "need >=2 sensor columns and a machine id"
        meta["features"] = feats
        return pd.DataFrame(), meta
    num = production[feats].apply(pd.to_numeric, errors="coerce")
    work = production[[machine]].copy()
    line = find_col(production, "line_id", "line")
    if line:
        work[line] = production[line]
    work[feats] = num
    work = work.dropna(subset=feats, how="all")
    if len(work) < 15:
        meta["reason"] = "need >=15 rows with sensors"
        meta["features"] = feats
        return pd.DataFrame(), meta
    X = work[feats].fillna(work[feats].median())
    iso = IsolationForest(contamination=0.08, random_state=42, n_estimators=100)
    labels = iso.fit_predict(X.values)
    scores = -iso.score_samples(X.values)
    work = work.copy()
    work["anomaly"] = labels == -1
    work["anomaly_score"] = scores
    g = (
        work.groupby(machine, dropna=False)
        .agg(
            sensor_rows=(feats[0], "count"),
            iso_anomalies=("anomaly", "sum"),
            iso_score_mean=("anomaly_score", "mean"),
        )
        .reset_index()
        .rename(columns={machine: "machine_id"})
    )
    g["iso_anomaly_rate_pct"] = (g["iso_anomalies"] / g["sensor_rows"].clip(lower=1)) * 100.0
    meta = {
        "ok": True,
        "features": feats,
        "anomaly_count": int(work["anomaly"].sum()),
        "rows": int(len(work)),
    }
    return g, meta


def remaining_risk_ranking(
    downtime: pd.DataFrame,
    production: Optional[pd.DataFrame] = None,
    top_n: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rule-based remaining risk plus optional IsolationForest sensor boost."""
    risk = asset_risk_table(downtime)
    iso_df, iso_meta = sensor_anomaly_by_asset(production)
    if risk.empty:
        return risk, iso_meta
    if frame_ok(iso_df):
        risk = risk.merge(iso_df, on="machine_id", how="left")
        risk["iso_anomaly_rate_pct"] = risk["iso_anomaly_rate_pct"].fillna(0)
        risk["iso_boost"] = (risk["iso_anomaly_rate_pct"] * 1.5).clip(0, 25)
    else:
        risk["iso_anomaly_rate_pct"] = 0.0
        risk["iso_boost"] = 0.0

    date_col = find_col(downtime, "shift_date", "start_time", "date")
    recent = _window(downtime, date_col, days=7)
    machine = find_col(downtime, "machine_id", "machine", "asset_id")
    minutes = find_col(downtime, "downtime_minutes", "downtime_min", "duration_min")
    if machine and minutes and frame_ok(recent):
        rec = (
            recent.groupby(machine)[minutes]
            .agg(events_7d="count", minutes_7d="sum")
            .reset_index()
            .rename(columns={machine: "machine_id"})
        )
        risk = risk.merge(rec, on="machine_id", how="left")
    else:
        risk["events_7d"] = risk["events"]
        risk["minutes_7d"] = risk["downtime_minutes"]
    risk["events_7d"] = pd.to_numeric(risk.get("events_7d"), errors="coerce").fillna(0)
    risk["minutes_7d"] = pd.to_numeric(risk.get("minutes_7d"), errors="coerce").fillna(0)
    recency = 10.0 * _norm01(risk["events_7d"])
    risk["remaining_risk"] = (risk["rule_score"] + risk["iso_boost"] + recency).clip(0, 100)

    def _why(r: pd.Series) -> str:
        bits = [
            f"MTTR {r['mttr_min']:.0f} min",
            f"{r['failure_freq_per_day']:.2f} fails/day",
            f"{r['unplanned_share']*100:.0f}% unplanned",
        ]
        if float(r.get("iso_anomaly_rate_pct") or 0) > 5:
            bits.append(f"sensor anomalies {r['iso_anomaly_rate_pct']:.0f}%")
        if float(r.get("events_7d") or 0) > 0:
            bits.append(f"{int(r['events_7d'])} events in last 7 days of data")
        return "; ".join(bits)

    risk["why"] = risk.apply(_why, axis=1)
    risk["action"] = np.where(
        risk["remaining_risk"] >= 55,
        "Inspect this week — kit spares and schedule a window",
        np.where(
            risk["remaining_risk"] >= 35,
            "Watch this week — add to walk-around list",
            "Routine PM cadence",
        ),
    )
    ranked = risk.sort_values("remaining_risk", ascending=False).head(top_n)
    return ranked, iso_meta


def availability_loss_hours(
    downtime: Optional[pd.DataFrame],
    oee_frame: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Tie downtime hours to the OEE Availability-loss narrative."""
    out: dict[str, Any] = {"ok": False}
    dt_hours = 0.0
    unplanned_hours = 0.0
    planned_dt_hours = 0.0
    if frame_ok(downtime):
        pu = planned_vs_unplanned(downtime)
        if pu.get("ok"):
            dt_hours = float(pu["total_min"]) / 60.0
            unplanned_hours = float(pu["unplanned_min"]) / 60.0
            planned_dt_hours = float(pu["planned_min"]) / 60.0
            out["class_source"] = pu.get("source")

    avail = None
    planned_prod_hours = None
    oee_loss_hours = None
    if frame_ok(oee_frame):
        try:
            summary = oee_summary(oee_frame)
            plant = summary["plant"]
            avail = float(plant.get("availability", 0))
            planned_col = find_col(oee_frame, "planned_time_min", "planned_minutes")
            if planned_col:
                planned_prod_hours = float(
                    pd.to_numeric(oee_frame[planned_col], errors="coerce").fillna(0).sum()
                ) / 60.0
            elif planned_prod_hours is None:
                work = summary.get("frame")
                if frame_ok(work) and "planned_time_min" in work.columns:
                    planned_prod_hours = float(work["planned_time_min"].sum()) / 60.0
            if avail is not None and planned_prod_hours:
                oee_loss_hours = (1.0 - avail) * planned_prod_hours
        except Exception:
            pass

    if dt_hours <= 0 and not oee_loss_hours:
        out["reason"] = "No downtime minutes or OEE planned time to convert to hours."
        return out

    hours_to_90 = None
    if avail is not None and planned_prod_hours:
        hours_to_90 = max(0.0, (0.90 - avail) * planned_prod_hours)

    narrative = (
        f"Downtime events account for {dt_hours:.1f} hours lost "
        f"({unplanned_hours:.1f} unplanned, {planned_dt_hours:.1f} planned)."
    )
    if avail is not None and oee_loss_hours is not None:
        narrative = (
            f"Plant Availability is {avail*100:.1f}%. Against "
            f"{planned_prod_hours:.0f} planned hours in scope, that is "
            f"{oee_loss_hours:.1f} hours of Availability loss on the OEE waterfall. "
            f"The downtime log explains {dt_hours:.1f} hours "
            f"({unplanned_hours:.1f} unplanned). "
        )
        if hours_to_90 and hours_to_90 > 0:
            narrative += (
                f"Closing Availability to 90% would recover about {hours_to_90:.1f} hours "
                "this period — the same hours the schedule is currently missing."
            )

    out.update(
        {
            "ok": True,
            "downtime_hours": round(dt_hours, 2),
            "unplanned_hours": round(unplanned_hours, 2),
            "planned_downtime_hours": round(planned_dt_hours, 2),
            "oee_availability": None if avail is None else round(avail, 4),
            "planned_production_hours": None
            if planned_prod_hours is None
            else round(planned_prod_hours, 2),
            "availability_loss_hours": None if oee_loss_hours is None else round(oee_loss_hours, 2),
            "hours_to_90pct_availability": None if hours_to_90 is None else round(hours_to_90, 2),
            "narrative": narrative.strip(),
        }
    )
    return out


def analyze_maintenance(
    downtime: Optional[pd.DataFrame],
    production: Optional[pd.DataFrame] = None,
    oee_frame: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Full maintenance overlay used by the page, insights, and reports."""
    if not frame_ok(downtime):
        return {"ok": False, "reason": "Downtime events table missing. Upload events or load sample plant data."}

    prod = _first_frame(production, oee_frame)
    oee_src = _first_frame(oee_frame, production)
    pu = planned_vs_unplanned(downtime)
    reliability = mttr_mtbf(downtime)
    try:
        pareto = downtime_pareto(downtime, top_n=10)
    except Exception:
        pareto = pd.DataFrame()
    by_machine = reason_pareto_by(downtime, by="machine_id")
    by_line = reason_pareto_by(downtime, by="line_id")
    longest = longest_events(downtime)
    assets = asset_risk_table(downtime)
    inspect, iso_meta = remaining_risk_ranking(downtime, production=prod)
    hours = availability_loss_hours(downtime, oee_src)

    cards: list[dict[str, Any]] = []
    if frame_ok(inspect):
        top = inspect.iloc[0]
        cards.append(
            {
                "priority": "high",
                "title": f"Inspect this week: {top.get('machine_id')}",
                "message": (
                    f"{top.get('machine_id')} on {top.get('line_id')} scores remaining-risk "
                    f"{float(top['remaining_risk']):.0f}/100. {top.get('why')}. {top.get('action')}."
                ),
            }
        )
    if pu.get("ok") and float(pu.get("unplanned_pct") or 0) >= 0.45:
        cards.append(
            {
                "priority": "high",
                "title": "Unplanned downtime dominates",
                "message": (
                    f"Unplanned stops are {float(pu['unplanned_pct'])*100:.0f}% of downtime minutes "
                    f"({float(pu['unplanned_min']):.0f} min). Treat changeover/PM as planned; "
                    "attack breakdowns with RCA and kitted spares."
                ),
            }
        )
    if hours.get("ok"):
        cards.append(
            {
                "priority": "medium",
                "title": "Availability loss in hours",
                "message": hours.get("narrative", ""),
            }
        )

    return {
        "ok": True,
        "reliability": reliability,
        "planned_vs_unplanned": {k: v for k, v in pu.items() if k != "classified"},
        "pareto": pareto,
        "reason_by_machine": by_machine,
        "reason_by_line": by_line,
        "longest_events": longest,
        "asset_risk": assets,
        "inspect_this_week": inspect,
        "iso": iso_meta,
        "hours_lost": hours,
        "cards": cards,
    }
