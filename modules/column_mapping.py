"""Saved column mapping: messy / SAP-like headers → canonical OEE Pulse fields."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Optional

import pandas as pd

from modules.session_store import _utcnow, connect, ensure_store

# Internal names expected by OEE / downtime / quality / maintenance engines.
CANONICAL_FIELDS: list[tuple[str, str, str]] = [
    ("line_id", "Line / work center", "identity"),
    ("machine_id", "Machine / asset / equipment", "identity"),
    ("shift", "Shift", "identity"),
    ("shift_date", "Shift / posting date", "datetime"),
    ("timestamp", "Event timestamp", "datetime"),
    ("planned_time_min", "Planned minutes", "numeric"),
    ("run_time_min", "Run minutes", "numeric"),
    ("ideal_rate", "Ideal rate (units/min)", "numeric"),
    ("total_count", "Total units produced", "numeric"),
    ("good_count", "Good units", "numeric"),
    ("reject_count", "Scrap / rejects", "numeric"),
    ("downtime_minutes", "Downtime minutes", "numeric"),
    ("downtime_code", "Downtime reason code", "identity"),
    ("downtime_category", "Downtime reason text", "identity"),
    ("start_time", "Event start", "datetime"),
    ("end_time", "Event end", "datetime"),
    ("is_planned", "Planned flag", "identity"),
    ("event_class", "Planned / unplanned class", "identity"),
    ("vibration_rms", "Vibration RMS", "numeric"),
    ("temp_c", "Temperature", "numeric"),
    ("motor_current_a", "Motor current", "numeric"),
    ("scrap_rate", "Scrap rate", "numeric"),
    ("defect_code", "Defect code", "identity"),
    ("defect_type", "Defect type", "identity"),
    ("measurement_mm", "Quality measurement", "numeric"),
    ("lsl", "Lower spec limit", "numeric"),
    ("usl", "Upper spec limit", "numeric"),
]

CANONICAL_NAMES = [c[0] for c in CANONICAL_FIELDS]
TABLE_KINDS = ("production", "downtime", "quality")

# SAP PP / PM / QM-ish aliases plus messy plant-export names.
ALIASES: dict[str, tuple[str, ...]] = {
    "line_id": (
        "line_id",
        "line",
        "prod_line",
        "production_line",
        "work_center",
        "workcenter",
        "arbpl",
        "arbpl_id",
        "crhd",
        "fehlort",
        "line_code",
    ),
    "machine_id": (
        "machine_id",
        "machine",
        "asset_id",
        "asset",
        "equipment",
        "equnr",
        "equnr_id",
        "tplnr",
        "func_loc",
        "functional_location",
        "asset_tag",
    ),
    "shift": ("shift", "shift_id", "shift_code", "schicht", "shift_name"),
    "shift_date": (
        "shift_date",
        "date",
        "prod_date",
        "production_date",
        "budat",
        "posting_date",
        "qmdat",
        "isdd",
        "proddate",
    ),
    "timestamp": ("timestamp", "datetime", "event_ts", "recorded_at"),
    "planned_time_min": (
        "planned_time_min",
        "planned_minutes",
        "planned_min",
        "available_time_min",
        "shift_minutes",
        "vgw02",
        "bezug",
        "plan_min",
    ),
    "run_time_min": (
        "run_time_min",
        "run_minutes",
        "runtime",
        "run_min",
        "istdz",
        "actual_run_min",
        "machine_time_min",
    ),
    "ideal_rate": ("ideal_rate", "ideal_rate_per_min", "design_rate", "target_rate", "cycletime_inv"),
    "total_count": (
        "total_count",
        "total_units",
        "produced_count",
        "output_count",
        "gamng",
        "menge",
        "qty_total",
    ),
    "good_count": (
        "good_count",
        "good_units",
        "good_output",
        "yield_qty",
        "lmnga",
        "yield",
        "good_qty",
    ),
    "reject_count": (
        "reject_count",
        "rejects",
        "scrap_count",
        "scrap",
        "xmnga",
        "fehleranzahl",
        "scrap_qty",
        "reject_qty",
    ),
    "downtime_minutes": (
        "downtime_minutes",
        "downtime_min",
        "duration_min",
        "auszt",
        "ausfallzeit",
        "down_min",
        "stop_minutes",
    ),
    "downtime_code": (
        "downtime_code",
        "reason_code",
        "failure_code",
        "fecod",
        "urcod",
        "ausfallursache",
        "stop_code",
    ),
    "downtime_category": (
        "downtime_category",
        "category",
        "reason",
        "qmtxt",
        "failure_text",
        "downtime_reason",
        "description",
    ),
    "start_time": ("start_time", "event_start", "ausvn", "start_ts"),
    "end_time": ("end_time", "event_end", "ausbs", "end_ts"),
    "is_planned": ("is_planned", "planned_flag", "planned"),
    "event_class": ("event_class", "event_type", "downtime_type", "stop_type"),
    "vibration_rms": ("vibration_rms", "vibration", "vib_rms", "schwingung"),
    "temp_c": ("temp_c", "temp", "temperature", "temperatur"),
    "motor_current_a": ("motor_current_a", "current", "amps", "motor_amps"),
    "scrap_rate": ("scrap_rate", "reject_rate", "scrap_pct"),
    "defect_code": ("defect_code", "defect", "qm_code", "fehlercode"),
    "defect_type": ("defect_type", "defect_desc", "fehlerart"),
    "measurement_mm": ("measurement_mm", "messwert", "measured_value", "reading"),
    "lsl": ("lsl", "tolunter", "lower_spec"),
    "usl": ("usl", "tolober", "upper_spec"),
}

SKIP_SOURCE = "(leave unmapped)"


def _norm(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("ß", "ss")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _score(source: str, target: str, aliases: tuple[str, ...]) -> float:
    src = _norm(source)
    tgt = _norm(target)
    best = SequenceMatcher(None, src, tgt).ratio()
    if src == tgt:
        return 1.0
    for alias in aliases:
        a = _norm(alias)
        if src == a:
            return 1.0
        if src.endswith("_" + a) or a in src.split("_") or src in a:
            best = max(best, 0.92)
        best = max(best, SequenceMatcher(None, src, a).ratio())
    return best


def suggest_mapping(columns: list[str], min_score: float = 0.62) -> dict[str, str]:
    """Map canonical field → source column (best unique match)."""
    remaining = list(columns)
    mapping: dict[str, str] = {}
    ranked: list[tuple[float, str, str]] = []
    for canon, _, _ in CANONICAL_FIELDS:
        aliases = ALIASES.get(canon, (canon,))
        for src in columns:
            ranked.append((_score(src, canon, aliases), canon, src))
    ranked.sort(key=lambda t: -t[0])
    used_src: set[str] = set()
    used_tgt: set[str] = set()
    for score, canon, src in ranked:
        if score < min_score or canon in used_tgt or src in used_src:
            continue
        mapping[canon] = src
        used_src.add(src)
        used_tgt.add(canon)
        if src in remaining:
            remaining.remove(src)
    return mapping


def mapping_confidence(columns: list[str], mapping: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for canon, src in mapping.items():
        if not src or src == SKIP_SOURCE:
            continue
        aliases = ALIASES.get(canon, (canon,))
        out[canon] = round(_score(src, canon, aliases), 3)
    return out


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename mapped columns to canonical names and derive run/downtime if needed.

    Does not use DataFrame truthiness. Unmapped columns are kept.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if df.empty:
        return df.copy()

    out = df.copy()
    rename: dict[str, str] = {}
    for canon, src in (mapping or {}).items():
        if not src or src == SKIP_SOURCE or src not in out.columns:
            continue
        if canon == src:
            continue
        if canon in out.columns and src != canon:
            # Keep original canonical; copy mapped source into a temp then overwrite
            out[canon] = out[src]
        else:
            rename[src] = canon
    if rename:
        out = out.rename(columns=rename)

    planned = "planned_time_min" if "planned_time_min" in out.columns else None
    run = "run_time_min" if "run_time_min" in out.columns else None
    down = "downtime_minutes" if "downtime_minutes" in out.columns else None
    if planned and run and down is None:
        p = pd.to_numeric(out[planned], errors="coerce")
        r = pd.to_numeric(out[run], errors="coerce")
        out["downtime_minutes"] = (p - r).clip(lower=0).fillna(0)
    if planned and down and run is None:
        p = pd.to_numeric(out[planned], errors="coerce")
        d = pd.to_numeric(out[down], errors="coerce")
        out["run_time_min"] = (p - d).clip(lower=0).fillna(0)

    for canon, _, kind in CANONICAL_FIELDS:
        if canon not in out.columns:
            continue
        if kind == "numeric":
            out[canon] = pd.to_numeric(out[canon], errors="coerce")
        elif kind == "datetime":
            out[canon] = pd.to_datetime(out[canon], errors="coerce")
    return out


def ensure_mapping_table() -> None:
    ensure_store()
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS column_mappings (
                plant_name TEXT NOT NULL,
                table_kind TEXT NOT NULL,
                session_id TEXT,
                mapping_json TEXT NOT NULL DEFAULT '{}',
                source_columns_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (plant_name, table_kind)
            )
            """
        )
        conn.commit()


def save_mapping(
    plant_name: str,
    table_kind: str,
    mapping: dict[str, str],
    *,
    session_id: Optional[str] = None,
    source_columns: Optional[list[str]] = None,
) -> None:
    ensure_mapping_table()
    plant = (plant_name or "Plant").strip() or "Plant"
    kind = (table_kind or "production").strip().lower()
    if kind not in TABLE_KINDS:
        kind = "production"
    payload = json.dumps(mapping or {}, ensure_ascii=False)
    cols = json.dumps(list(source_columns or []), ensure_ascii=False)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO column_mappings(plant_name, table_kind, session_id, mapping_json, source_columns_json, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(plant_name, table_kind) DO UPDATE SET
              session_id=excluded.session_id,
              mapping_json=excluded.mapping_json,
              source_columns_json=excluded.source_columns_json,
              updated_at=excluded.updated_at
            """,
            (plant, kind, session_id, payload, cols, _utcnow()),
        )
        conn.commit()


def load_mapping(plant_name: str, table_kind: str) -> Optional[dict[str, str]]:
    ensure_mapping_table()
    plant = (plant_name or "Plant").strip() or "Plant"
    kind = (table_kind or "production").strip().lower()
    with connect() as conn:
        row = conn.execute(
            "SELECT mapping_json FROM column_mappings WHERE plant_name=? AND table_kind=?",
            (plant, kind),
        ).fetchone()
    if row is None:
        return None
    try:
        data = json.loads(row["mapping_json"] or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {str(k): str(v) for k, v in data.items() if v}


def list_saved_mappings(plant_name: Optional[str] = None) -> list[dict[str, Any]]:
    ensure_mapping_table()
    with connect() as conn:
        if plant_name:
            rows = conn.execute(
                "SELECT plant_name, table_kind, session_id, mapping_json, updated_at "
                "FROM column_mappings WHERE plant_name=? ORDER BY table_kind",
                (plant_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT plant_name, table_kind, session_id, mapping_json, updated_at "
                "FROM column_mappings ORDER BY updated_at DESC"
            ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            mapping = json.loads(r["mapping_json"] or "{}")
        except json.JSONDecodeError:
            mapping = {}
        out.append(
            {
                "plant_name": r["plant_name"],
                "table_kind": r["table_kind"],
                "session_id": r["session_id"],
                "mapping": mapping,
                "updated_at": r["updated_at"],
            }
        )
    return out


def resolve_mapping(columns: list[str], plant_name: str, table_kind: str) -> dict[str, str]:
    """Prefer a saved mapping whose source columns overlap; else suggest."""
    saved = load_mapping(plant_name, table_kind)
    cols = set(columns)
    if saved:
        hits = sum(1 for src in saved.values() if src and src != SKIP_SOURCE and src in cols)
        if hits >= 2 or (hits >= 1 and len(saved) <= 4):
            return saved
    return suggest_mapping(columns)


def apply_saved_or_suggested(
    df: pd.DataFrame,
    plant_name: str,
    table_kind: str,
) -> tuple[pd.DataFrame, dict[str, str], str]:
    """Apply saved mapping when source columns match, otherwise high-confidence suggestions."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        empty = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        return empty, {}, "empty"
    mapping = resolve_mapping(list(df.columns), plant_name, table_kind)
    saved = load_mapping(plant_name, table_kind)
    how = "saved" if saved and mapping == saved else "suggested"
    return apply_mapping(df, mapping), mapping, how
