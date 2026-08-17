"""Quality department overlay: scrap, FPY, defect Pareto, SPC lite, Cp/Cpk."""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np
import pandas as pd

from modules.oee_engine import oee_summary
from modules.quality_checks import find_col

SYNTH_DEFECTS = ("DIM", "SCRATCH", "CONTAM", "OTHER")


def frame_ok(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty


def _first_frame(*frames: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    for f in frames:
        if frame_ok(f):
            return f
    return None


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


def _ensure_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    work = df.copy()
    good = find_col(work, "good_count", "good_units", "good_output")
    rejects = find_col(work, "reject_count", "rejects", "scrap_count", "defect_count")
    total = find_col(work, "total_count", "total_units", "produced_count")
    scrap = find_col(work, "scrap_rate", "reject_rate")

    if total is None and good and rejects:
        work["total_count"] = pd.to_numeric(work[good], errors="coerce").fillna(0) + pd.to_numeric(
            work[rejects], errors="coerce"
        ).fillna(0)
        total = "total_count"
    if good is None and total and rejects:
        work["good_count"] = (
            pd.to_numeric(work[total], errors="coerce").fillna(0)
            - pd.to_numeric(work[rejects], errors="coerce").fillna(0)
        ).clip(lower=0)
        good = "good_count"
    if rejects is None and total and good:
        work["reject_count"] = (
            pd.to_numeric(work[total], errors="coerce").fillna(0)
            - pd.to_numeric(work[good], errors="coerce").fillna(0)
        ).clip(lower=0)
        rejects = "reject_count"
    if total is None:
        work["total_count"] = 0.0
        total = "total_count"
    if good is None:
        work["good_count"] = 0.0
        good = "good_count"
    if rejects is None:
        work["reject_count"] = 0.0
        rejects = "reject_count"

    for c in (good, rejects, total):
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)
    t = work[total].replace(0, np.nan)
    computed = (work[rejects] / t).fillna(0).clip(0, 1)
    if scrap:
        work[scrap] = pd.to_numeric(work[scrap], errors="coerce")
        work["scrap_rate"] = work[scrap].fillna(computed)
    else:
        work["scrap_rate"] = computed
    work["fpy"] = (1.0 - work["scrap_rate"]).clip(0, 1)
    return work, good, rejects, total


def ensure_defect_codes(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Use defect_code/type if present; otherwise synthesize a stable code from the row."""
    work = df.copy()
    code = find_col(work, "defect_code", "defect_type", "defect", "reject_code", "nc_code")
    if code:
        work["defect_code"] = work[code].astype(str).replace({"nan": "UNSPECIFIED", "None": "UNSPECIFIED"})
        work["defect_code"] = work["defect_code"].fillna("UNSPECIFIED")
        return work, False
    synthesized = True
    machine = find_col(work, "machine_id", "machine")
    line = find_col(work, "line_id", "line")
    seed = pd.Series([""] * len(work), index=work.index)
    if machine:
        seed = seed + work[machine].astype(str)
    if line:
        seed = seed + work[line].astype(str)
    def _bucket(s: str) -> str:
        h = int(hashlib.md5(str(s).encode("utf-8")).hexdigest(), 16)
        return SYNTH_DEFECTS[h % len(SYNTH_DEFECTS)]

    idx = seed.map(_bucket)
    work["defect_code"] = idx
    return work, synthesized


def scrap_and_fpy(
    quality: pd.DataFrame,
    oee_frame: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    work, good_c, rej_c, tot_c = _ensure_counts(quality)
    good = float(work[good_c].sum())
    rejects = float(work[rej_c].sum())
    total = float(work[tot_c].sum())
    if total <= 0:
        total = good + rejects
    scrap = (rejects / total) if total else 0.0
    fpy = 1.0 - scrap if total else 0.0
    quality_oee = None
    quality_loss = None
    if frame_ok(oee_frame):
        try:
            plant = oee_summary(oee_frame)["plant"]
            quality_oee = float(plant.get("quality", 0))
            a = float(plant.get("availability", 0))
            p = float(plant.get("performance", 0))
            quality_loss = a * p * (1.0 - quality_oee)
        except Exception:
            quality_oee = fpy
    else:
        quality_oee = fpy

    line = find_col(work, "line_id", "line")
    by_line = pd.DataFrame()
    if line:
        by_line = (
            work.groupby(line, dropna=False)
            .agg(
                good=(good_c, "sum"),
                rejects=(rej_c, "sum"),
                total=(tot_c, "sum"),
                scrap_rate=("scrap_rate", "mean"),
                fpy=("fpy", "mean"),
            )
            .reset_index()
            .rename(columns={line: "line_id"})
        )
        by_line["scrap_rate_weighted"] = np.where(
            by_line["total"] > 0, by_line["rejects"] / by_line["total"], by_line["scrap_rate"]
        )
        by_line["fpy_weighted"] = 1.0 - by_line["scrap_rate_weighted"]
        by_line["reject_share"] = by_line["rejects"] / max(rejects, 1.0)

    return {
        "ok": True,
        "good": round(good, 1),
        "rejects": round(rejects, 1),
        "total": round(total, 1),
        "scrap_rate": round(scrap, 4),
        "fpy": round(fpy, 4),
        "quality_oee": None if quality_oee is None else round(quality_oee, 4),
        "quality_loss_oee_pts": None if quality_loss is None else round(quality_loss, 4),
        "by_line": by_line,
    }


def defect_pareto(quality: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, bool]:
    if not frame_ok(quality):
        return pd.DataFrame(), False
    work, synthesized = ensure_defect_codes(quality)
    work, _good_c, rej, tot_c = _ensure_counts(work)
    g = (
        work.groupby("defect_code", dropna=False)[rej]
        .agg(events="count", rejects="sum")
        .reset_index()
        .sort_values("rejects", ascending=False)
    )
    total = float(g["rejects"].sum()) or 1.0
    g["pct"] = g["rejects"] / total
    g["cum_pct"] = g["pct"].cumsum()
    g = g.head(top_n)
    g["synthesized"] = synthesized
    return g, synthesized


def spc_lite(quality: pd.DataFrame) -> dict[str, Any]:
    """X-bar / moving-range control limits on scrap % (fallback: reject count) per shift/line."""
    if not frame_ok(quality):
        return {"ok": False, "reason": "Quality / rejects table missing."}
    work, _good_c, rej_c, _tot_c = _ensure_counts(quality)
    date_col = find_col(work, "shift_date", "date", "timestamp")
    line = find_col(work, "line_id", "line")
    shift = find_col(work, "shift")
    metric = "scrap_rate"
    label = "scrap_rate"
    if work[metric].notna().sum() < 5:
        metric = rej_c
        label = "reject_count"

    group_keys = [c for c in (date_col, shift, line) if c]
    if not group_keys:
        return {"ok": False, "reason": "Need shift_date / shift / line_id to build an SPC series."}

    grouped = work.groupby(group_keys, dropna=False, as_index=False)[metric].mean()
    grouped = grouped.rename(columns={metric: "xbar"})
    if date_col:
        grouped[date_col] = pd.to_datetime(grouped[date_col], errors="coerce")
        grouped = grouped.sort_values([c for c in (date_col, shift, line) if c in grouped.columns])

    def _imr(series: pd.Series) -> dict[str, Any]:
        x = pd.to_numeric(series, errors="coerce").dropna()
        if len(x) < 5:
            return {"ok": False, "reason": "Need >=5 subgroups for control limits."}
        mr = x.diff().abs().dropna()
        mr_bar = float(mr.mean()) if len(mr) else 0.0
        cl = float(x.mean())
        ucl = cl + 2.66 * mr_bar
        lcl = max(0.0, cl - 2.66 * mr_bar)
        pts = x.reset_index(drop=True)
        ooc = [
            {"index": int(i), "value": float(v)}
            for i, v in pts.items()
            if v > ucl or v < lcl
        ]
        return {
            "ok": True,
            "n": int(len(x)),
            "cl": round(cl, 5),
            "ucl": round(ucl, 5),
            "lcl": round(lcl, 5),
            "mr_bar": round(mr_bar, 5),
            "out_of_control": ooc,
            "ooc_count": len(ooc),
        }

    if date_col and date_col in grouped.columns:
        plant_series = grouped.groupby(date_col, as_index=False)["xbar"].mean().sort_values(date_col)
    else:
        plant_series = grouped[["xbar"]].copy()
    plant_stats = _imr(plant_series["xbar"])
    plant_points = plant_series.copy()
    plant_points["point"] = range(1, len(plant_points) + 1)

    by_line: dict[str, Any] = {}
    line_frames: list[pd.DataFrame] = []
    if line and line in grouped.columns:
        for lid, g in grouped.groupby(line, dropna=False):
            stats = _imr(g["xbar"])
            pts = g.copy()
            pts["point"] = range(1, len(pts) + 1)
            pts["line_id"] = lid
            if stats.get("ok"):
                pts["cl"] = stats["cl"]
                pts["ucl"] = stats["ucl"]
                pts["lcl"] = stats["lcl"]
                pts["ooc"] = (pts["xbar"] > stats["ucl"]) | (pts["xbar"] < stats["lcl"])
            else:
                pts["ooc"] = False
            by_line[str(lid)] = {**stats, "points": pts}
            line_frames.append(pts)

    chart_df = pd.concat(line_frames, ignore_index=True) if line_frames else plant_points
    return {
        "ok": bool(plant_stats.get("ok") or any(v.get("ok") for v in by_line.values())),
        "method": "X-bar / moving range (I-MR on subgroup means)",
        "metric": label,
        "plant": {**plant_stats, "points": plant_points},
        "by_line": by_line,
        "chart_df": chart_df,
        "reason": None if plant_stats.get("ok") else plant_stats.get("reason"),
    }


def _spec_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str]]:
    meas = find_col(
        df,
        "measurement_mm",
        "measurement",
        "measured_value",
        "dimension",
        "thickness",
        "weight",
        "spec_value",
    )
    usl = find_col(df, "usl", "spec_usl", "upper_spec", "usl_mm")
    lsl = find_col(df, "lsl", "spec_lsl", "lower_spec", "lsl_mm")
    return meas, lsl, usl


def cpk_lite(quality: pd.DataFrame) -> dict[str, Any]:
    """Cp/Cpk when a numeric measurement plus LSL/USL exist; skip otherwise."""
    if not frame_ok(quality):
        return {"ok": False, "reason": "Quality table missing."}
    meas, lsl_c, usl_c = _spec_columns(quality)
    if not meas or not lsl_c or not usl_c:
        return {
            "ok": False,
            "reason": "No spec-like columns (need a measurement plus LSL and USL). Skipping Cp/Cpk.",
        }
    x = pd.to_numeric(quality[meas], errors="coerce").dropna()
    if len(x) < 10:
        return {"ok": False, "reason": "Need >=10 numeric measurements for Cp/Cpk."}
    lsl = pd.to_numeric(quality[lsl_c], errors="coerce").dropna()
    usl = pd.to_numeric(quality[usl_c], errors="coerce").dropna()
    if lsl.empty or usl.empty:
        return {"ok": False, "reason": "LSL/USL could not be parsed."}
    lo, hi = float(lsl.median()), float(usl.median())
    if hi <= lo:
        return {"ok": False, "reason": "Invalid spec window (USL <= LSL)."}
    mu = float(x.mean())
    sigma = float(x.std(ddof=1))
    if sigma <= 1e-12:
        return {"ok": False, "reason": "Measurement has no variation."}
    cp = (hi - lo) / (6.0 * sigma)
    cpk = min((hi - mu) / (3.0 * sigma), (mu - lo) / (3.0 * sigma))
    by_line = pd.DataFrame()
    line = find_col(quality, "line_id", "line")
    if line:
        rows = []
        for lid, g in quality.groupby(line, dropna=False):
            xx = pd.to_numeric(g[meas], errors="coerce").dropna()
            if len(xx) < 8:
                continue
            s = float(xx.std(ddof=1))
            if s <= 1e-12:
                continue
            m = float(xx.mean())
            rows.append(
                {
                    "line_id": lid,
                    "n": int(len(xx)),
                    "mean": round(m, 4),
                    "std": round(s, 4),
                    "cp": round((hi - lo) / (6.0 * s), 3),
                    "cpk": round(min((hi - m) / (3.0 * s), (m - lo) / (3.0 * s)), 3),
                }
            )
        by_line = pd.DataFrame(rows).sort_values("cpk") if rows else pd.DataFrame()
    return {
        "ok": True,
        "measurement": meas,
        "lsl": round(lo, 4),
        "usl": round(hi, 4),
        "mean": round(mu, 4),
        "std": round(sigma, 4),
        "n": int(len(x)),
        "cp": round(cp, 3),
        "cpk": round(cpk, 3),
        "by_line": by_line,
    }


def quality_line_cards(
    quality: pd.DataFrame,
    oee_frame: Optional[pd.DataFrame] = None,
) -> list[dict[str, Any]]:
    """Which line is killing quality this week."""
    cards: list[dict[str, Any]] = []
    if not frame_ok(quality):
        return cards
    date_col = find_col(quality, "shift_date", "date", "timestamp")
    week = _window(quality, date_col, days=7)
    stats = scrap_and_fpy(week, oee_frame)
    by_line = stats.get("by_line")
    if not isinstance(by_line, pd.DataFrame) or by_line.empty:
        return cards
    worst = by_line.sort_values("scrap_rate_weighted", ascending=False).iloc[0]
    best = by_line.sort_values("scrap_rate_weighted", ascending=True).iloc[0]
    plant_scrap = float(stats.get("scrap_rate") or 0)
    share = float(worst.get("reject_share") or 0)
    cards.append(
        {
            "priority": "high",
            "title": f"Line {worst['line_id']} is killing quality this week",
            "message": (
                f"Line {worst['line_id']} scrap is {float(worst['scrap_rate_weighted'])*100:.1f}% "
                f"(FPY {float(worst['fpy_weighted'])*100:.1f}%) vs plant "
                f"{plant_scrap*100:.1f}%. It contributes {share*100:.0f}% of rejects. "
                f"Best line {best['line_id']} is at {float(best['scrap_rate_weighted'])*100:.1f}% scrap. "
                "Contain there before spreading the Quality-loss kaizen."
            ),
        }
    )
    if frame_ok(oee_frame):
        try:
            by = oee_summary(oee_frame).get("by_line")
            if isinstance(by, pd.DataFrame) and not by.empty and "quality" in by.columns:
                qworst = by.sort_values("quality").iloc[0]
                cards.append(
                    {
                        "priority": "medium",
                        "title": "Quality component of OEE",
                        "message": (
                            f"OEE Quality is weakest on line {qworst['line_id']} at "
                            f"{float(qworst['quality'])*100:.1f}%. Scrap on the quality log is the "
                            "same lever — every reject point hits the Q in A×P×Q."
                        ),
                    }
                )
        except Exception:
            pass
    return cards


def analyze_quality(
    quality: Optional[pd.DataFrame],
    oee_frame: Optional[pd.DataFrame] = None,
    production: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    src = _first_frame(quality, oee_frame, production)
    if not frame_ok(src):
        return {
            "ok": False,
            "reason": "Quality / rejects table missing. Upload rejects or load sample plant data.",
        }
    has_q_signal = any(
        find_col(src, n)
        for n in ("good_count", "reject_count", "scrap_rate", "defect_code", "defect_type")
    )
    if quality is None or (isinstance(quality, pd.DataFrame) and quality.empty):
        if not has_q_signal:
            return {
                "ok": False,
                "reason": "Quality / rejects table missing. Upload rejects or load sample plant data.",
            }

    kpis = scrap_and_fpy(src, oee_frame if frame_ok(oee_frame) else production)
    pareto, synthesized = defect_pareto(src)
    spc = spc_lite(src)
    cpk = cpk_lite(src)
    cards = quality_line_cards(src, oee_frame if frame_ok(oee_frame) else production)

    if synthesized:
        cards.append(
            {
                "priority": "low",
                "title": "Defect codes synthesized",
                "message": (
                    "No defect_code/type column on the quality table — Pareto uses synthesized "
                    "codes so the chart still runs. Upload a defect code column for a true Pareto."
                ),
            }
        )
    if spc.get("ok"):
        ooc = int(spc.get("plant", {}).get("ooc_count") or 0)
        if ooc:
            cards.append(
                {
                    "priority": "high",
                    "title": "SPC: points outside control limits",
                    "message": (
                        f"{ooc} plant-level subgroup(s) sit outside I-MR limits on "
                        f"{spc.get('metric')}. Investigate those shifts before tweaking specs."
                    ),
                }
            )
    if cpk.get("ok") and float(cpk.get("cpk") or 99) < 1.0:
        cards.append(
            {
                "priority": "high",
                "title": f"Cpk {cpk['cpk']} — process not capable",
                "message": (
                    f"{cpk['measurement']} Cpk={cpk['cpk']} (Cp={cpk['cp']}) vs LSL {cpk['lsl']} / "
                    f"USL {cpk['usl']}. Center and tighten before chasing more OEE Quality points."
                ),
            }
        )

    order = {"high": 0, "medium": 1, "low": 2}
    cards.sort(key=lambda x: order.get(x.get("priority"), 9))
    return {
        "ok": True,
        "kpis": kpis,
        "defect_pareto": pareto,
        "defect_synthesized": synthesized,
        "spc": spc,
        "cpk": cpk,
        "cards": cards,
        "used_fallback_frame": not frame_ok(quality),
    }
