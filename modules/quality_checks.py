"""Industrial data-quality checks adapted for OEE / plant production data."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def find_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for n in names:
        for k, real in lower.items():
            if n.lower() in k:
                return real
    return None


def _zscore_iqr_flags(df: pd.DataFrame) -> dict[str, Any]:
    num = df.select_dtypes(include=[np.number])
    z_hits, iqr_hits = 0, 0
    details: list[str] = []
    for c in num.columns:
        s = num[c].dropna()
        if len(s) < 5:
            continue
        z = (s - s.mean()) / (s.std() + 1e-9)
        zc = int((z.abs() > 3).sum())
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        ic = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr else 0
        z_hits += zc
        iqr_hits += ic
        if zc or ic:
            details.append(f"{c}:z={zc},iqr={ic}")
    return {"z_hits": z_hits, "iqr_hits": iqr_hits, "details": details[:8]}


def _isolation_forest_flags(df: pd.DataFrame) -> dict[str, Any]:
    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 15:
        return {"ok": False, "reason": "need >=2 numeric cols & 15 rows"}
    iso = IsolationForest(contamination=0.08, random_state=42)
    labels = iso.fit_predict(num.values)
    n = int((labels == -1).sum())
    return {"ok": True, "anomalies": n, "rate_pct": round(100.0 * n / len(num), 2)}


def _dbscan_noise(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 20:
        return {"ok": False, "reason": "need more numeric rows"}
    X = StandardScaler().fit_transform(num.values[: min(2000, len(num))])
    labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)
    noise = int((labels == -1).sum())
    return {
        "ok": True,
        "noise_points": noise,
        "clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
    }


def _kmeans_clean_proxy(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 20:
        return {"ok": False}
    X = StandardScaler().fit_transform(num.values[: min(2000, len(num))])
    km = KMeans(n_clusters=min(3, max(2, len(X) // 5)), random_state=42, n_init=10)
    labels = km.fit_predict(X)
    dists = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
    far = int((dists > dists.mean() + 2 * dists.std()).sum())
    return {"ok": True, "far_from_cluster": far}


def _oee_range_rules(df: pd.DataFrame) -> list[str]:
    """Domain rules for OEE plant columns."""
    flags: list[str] = []
    for name, lo, hi in [
        ("availability", 0, 1.5),
        ("performance", 0, 1.5),
        ("quality", 0, 1.5),
        ("oee", 0, 1.5),
        ("scrap_rate", 0, 1),
        ("ideal_rate", 0, None),
    ]:
        col = find_col(df, name)
        if not col:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if lo is not None and (s < lo).fillna(False).any():
            flags.append(f"{col}: values below {lo}")
        if hi is not None and (s > hi).fillna(False).any():
            flags.append(f"{col}: values above {hi}")

    dt = find_col(df, "downtime_minutes", "downtime_min")
    pt = find_col(df, "planned_time_min", "planned_minutes", "available_time_min")
    if dt and pt:
        d = pd.to_numeric(df[dt], errors="coerce")
        p = pd.to_numeric(df[pt], errors="coerce")
        over = int(((d > p) & p.notna() & d.notna()).sum())
        if over:
            flags.append(f"Downtime exceeds planned time in {over} rows")

    good = find_col(df, "good_count", "good_units")
    total = find_col(df, "total_count", "total_units", "produced_count")
    if good and total:
        g = pd.to_numeric(df[good], errors="coerce")
        t = pd.to_numeric(df[total], errors="coerce")
        bad = int(((g > t) & g.notna() & t.notna()).sum())
        if bad:
            flags.append(f"Good count > total count in {bad} rows")
    return flags


def _rolling_impossible_jumps(df: pd.DataFrame) -> dict[str, Any]:
    flags: list[str] = []
    total = 0
    watch = ("rate", "count", "minutes", "scrap", "oee", "speed", "output")
    for c in df.select_dtypes(include=[np.number]).columns:
        cl = str(c).lower()
        if not any(h in cl for h in watch):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        delta = s.diff().abs()
        thr = float(s.std() or 1) * 5
        n = int((delta > thr).fillna(False).sum())
        if n:
            flags.append(f"{c}:{n} jumps>{thr:.1f}")
            total += n
    return {"ok": True, "impossible_jumps": flags[:10], "count": total}


def _pca_drift(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 30:
        return {"ok": False}
    half = len(num) // 2
    X1 = StandardScaler().fit_transform(num.iloc[:half])
    X2 = StandardScaler().fit_transform(num.iloc[half:])
    ncomp = min(3, X1.shape[1])
    r1 = float(PCA(n_components=ncomp).fit(X1).explained_variance_ratio_.sum())
    r2 = float(PCA(n_components=min(3, X2.shape[1])).fit(X2).explained_variance_ratio_.sum())
    drift = abs(r1 - r2)
    return {
        "ok": True,
        "pca_var_early": round(r1, 3),
        "pca_var_late": round(r2, 3),
        "drift_score": round(drift, 3),
        "concept_drift": drift > 0.15,
    }


def _association_rules_proxy(df: pd.DataFrame) -> dict[str, Any]:
    try:
        items: list[set[str]] = []
        cats = [
            c
            for c in df.select_dtypes(include=["object", "category"]).columns
            if df[c].nunique(dropna=True) <= 12
        ][:4]
        nums = list(df.select_dtypes(include=[np.number]).columns)[:6]
        sample = df.tail(min(800, len(df)))
        for _, row in sample.iterrows():
            basket: set[str] = set()
            for c in cats:
                val = row[c]
                if pd.notna(val):
                    basket.add(f"{c}={val}")
            for c in nums:
                s = pd.to_numeric(sample[c], errors="coerce")
                thr = s.quantile(0.9)
                v = pd.to_numeric(row[c], errors="coerce")
                if pd.notna(v) and pd.notna(thr) and v >= thr:
                    basket.add(f"{c}=HIGH")
            if len(basket) >= 2:
                items.append(basket)
        if len(items) < 20:
            return {"ok": True, "skipped": "too few baskets", "suspicious_rules": []}
        pair_counts: Counter[tuple[str, str]] = Counter()
        item_counts: Counter[str] = Counter()
        for basket in items:
            for a in basket:
                item_counts[a] += 1
            bl = sorted(basket)
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)):
                    pair_counts[(bl[i], bl[j])] += 1
        n = len(items)
        rules = []
        for (a, b), cnt in pair_counts.most_common(30):
            support = cnt / n
            conf_ab = cnt / max(1, item_counts[a])
            conf_ba = cnt / max(1, item_counts[b])
            if support >= 0.05 and max(conf_ab, conf_ba) >= 0.55:
                rules.append(
                    {
                        "rule": f"{a} => {b}",
                        "support": round(support, 3),
                        "confidence": round(max(conf_ab, conf_ba), 3),
                    }
                )
        suspicious = [r for r in rules if "HIGH" in r["rule"] and r["confidence"] >= 0.7][:8]
        return {
            "ok": True,
            "rules_found": len(rules),
            "top_rules": rules[:5],
            "suspicious_rules": suspicious,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "suspicious_rules": []}


def clean_plant_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Basic industrial cleaning: trim strings, coerce numerics, drop empty rows."""
    out = df.copy()
    log: dict[str, Any] = {"rows_in": len(out)}
    out = out.dropna(how="all")
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    for c in out.columns:
        cl = str(c).lower()
        if any(
            k in cl
            for k in (
                "count",
                "minutes",
                "rate",
                "oee",
                "availability",
                "performance",
                "quality",
                "scrap",
                "ideal",
                "target",
                "produced",
            )
        ):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        if any(k in cl for k in ("timestamp", "date", "time")):
            out[c] = pd.to_datetime(out[c], errors="coerce")
    dups = int(out.duplicated().sum())
    out = out.drop_duplicates()
    log["duplicates_removed"] = dups
    log["rows_out"] = len(out)
    return out, log


def build_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Run industrial quality checks tailored for OEE plant data."""
    checks: list[dict[str, Any]] = []
    n, m = df.shape
    miss = float(df.isna().sum().sum() / max(1, df.size))
    checks.append(
        {
            "check": "NULLS / Missing%",
            "status": "FAIL" if miss > 0.2 else ("WARN" if miss > 0.05 else "PASS"),
            "detail": f"{miss * 100:.2f}% missing",
        }
    )
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    checks.append(
        {
            "check": "CONSTANT",
            "status": "FAIL" if const_cols else "PASS",
            "detail": str(const_cols[:6]) if const_cols else "none",
        }
    )
    num = df.select_dtypes(include=[np.number])
    zero_ratio = float((num == 0).sum().sum() / max(1, num.size)) if num.size else 0
    checks.append(
        {
            "check": "ZEROS",
            "status": "WARN" if zero_ratio > 0.3 else "PASS",
            "detail": f"{zero_ratio * 100:.1f}% zeros",
        }
    )
    dups = int(df.duplicated().sum())
    checks.append(
        {
            "check": "DUPLICATES",
            "status": "WARN" if dups else "PASS",
            "detail": f"{dups} dup rows",
        }
    )
    zi = _zscore_iqr_flags(df)
    checks.append(
        {
            "check": "Z-SCORE (>3σ)",
            "status": "WARN" if zi["z_hits"] else "PASS",
            "detail": f"{zi['z_hits']} hits; {zi['details'][:3]}",
        }
    )
    checks.append(
        {
            "check": "IQR OUTLIER",
            "status": "WARN" if zi["iqr_hits"] else "PASS",
            "detail": f"{zi['iqr_hits']} hits",
        }
    )
    iso = _isolation_forest_flags(df)
    checks.append(
        {
            "check": "ISOLATION FOREST",
            "status": "WARN" if iso.get("anomalies", 0) else ("PASS" if iso.get("ok") else "INFO"),
            "detail": json.dumps({k: iso[k] for k in iso if k != "ok"})[:160],
        }
    )
    db = _dbscan_noise(df)
    checks.append(
        {
            "check": "DBSCAN NOISE",
            "status": "WARN" if db.get("noise_points", 0) else ("PASS" if db.get("ok") else "INFO"),
            "detail": json.dumps(db)[:160],
        }
    )
    km = _kmeans_clean_proxy(df)
    checks.append(
        {
            "check": "KMEANS DISTANCE",
            "status": "WARN" if km.get("far_from_cluster", 0) else ("PASS" if km.get("ok") else "INFO"),
            "detail": json.dumps(km)[:160],
        }
    )
    jumps = _rolling_impossible_jumps(df)
    checks.append(
        {
            "check": "ROLLING IMPOSSIBLE JUMP",
            "status": "FAIL" if jumps.get("count", 0) else "PASS",
            "detail": str(jumps.get("impossible_jumps") or "none")[:160],
        }
    )
    domain_flags = _oee_range_rules(df)
    checks.append(
        {
            "check": "OEE DOMAIN RULES",
            "status": "FAIL" if domain_flags else "PASS",
            "detail": "; ".join(domain_flags) if domain_flags else "no domain violations",
        }
    )
    pca = _pca_drift(df)
    checks.append(
        {
            "check": "PCA / CONCEPT DRIFT",
            "status": "WARN" if pca.get("concept_drift") else ("PASS" if pca.get("ok") else "INFO"),
            "detail": json.dumps(pca)[:160],
        }
    )
    assoc = _association_rules_proxy(df)
    checks.append(
        {
            "check": "ASSOCIATION RULE MINING",
            "status": "WARN" if assoc.get("suspicious_rules") else ("PASS" if assoc.get("ok") else "INFO"),
            "detail": json.dumps(assoc)[:160],
        }
    )
    mid = find_col(df, "machine_id", "machine", "asset_id")
    checks.append(
        {
            "check": "MACHINE ID PRESENT",
            "status": "PASS" if mid else "WARN",
            "detail": str(mid or "missing"),
        }
    )
    checks.append(
        {
            "check": "TIMESTAMP / SHIFT DATE",
            "status": "PASS"
            if find_col(df, "timestamp", "shift_date", "date", "time", "datetime")
            else "WARN",
            "detail": str(
                find_col(df, "timestamp", "shift_date", "date", "time", "datetime") or "missing"
            ),
        }
    )
    checks.append(
        {
            "check": "SCHEMA / ROWCOUNT",
            "status": "PASS" if n > 0 else "FAIL",
            "detail": f"{n} rows × {m} cols",
        }
    )
    fail = sum(1 for c in checks if c["status"] == "FAIL")
    warn = sum(1 for c in checks if c["status"] == "WARN")
    return {
        "checks": checks,
        "summary": {"pass": len(checks) - fail - warn, "warn": warn, "fail": fail, "total": len(checks)},
        "pca": pca,
        "domain_flags": domain_flags,
        "association": assoc,
    }


QUALITY_STAGE_COUNT = 14
