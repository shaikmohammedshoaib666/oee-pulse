"""Multi-file plant data integration with SQL-style joins."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

JOIN_TYPES = {
    "inner": "INNER JOIN — only matching keys in both tables",
    "left": "LEFT JOIN — all rows from left + matches from right",
    "right": "RIGHT JOIN — all rows from right + matches from left",
    "outer": "FULL OUTER JOIN — all rows from both tables",
}

PathLike = Union[str, Path]
TABULAR_SUFFIXES = (".csv", ".tsv", ".xlsx", ".xls", ".xlsm", ".json", ".parquet")
TABLE_KINDS = ("production", "downtime", "quality")

_KIND_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "downtime": (
        "downtime",
        "down_time",
        "breakdown",
        "sap_pm",
        "_pm_",
        "notification",
        "unplanned",
    ),
    "quality": (
        "quality",
        "reject",
        "scrap",
        "sap_qm",
        "_qm_",
        "defect",
        "inspection",
    ),
    "production": (
        "production",
        "prod_log",
        "prodlog",
        "sap_pp",
        "_pp_",
        "yield",
        "output",
        "confirm",
    ),
}
_KIND_COLUMN_HINTS: dict[str, tuple[str, ...]] = {
    "downtime": ("downtime_minutes", "downtime_code", "event_id", "auszt", "fecod"),
    "quality": ("reject_count", "scrap_rate", "defect_code", "xmnga", "fehleranzahl"),
    "production": ("planned_time_min", "ideal_rate", "total_count", "vgw02", "lmnnga", "gamng"),
}


def load_tabular_file(uploaded_file) -> pd.DataFrame:
    """Load csv/tsv/xlsx/json into a DataFrame from a Streamlit UploadedFile or path."""
    name = getattr(uploaded_file, "name", str(uploaded_file)).lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if name.endswith(".parquet"):
        try:
            return pd.read_parquet(uploaded_file)
        except ImportError:
            import duckdb

            return duckdb.read_parquet(uploaded_file).df()
    if name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    df = pd.read_csv(uploaded_file)
    for col in df.columns:
        cl = str(col).lower()
        if cl in {"timestamp", "start_time", "end_time", "datetime", "shift_date"}:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_path(path: PathLike) -> pd.DataFrame:
    return load_tabular_file(Path(path))


def _zip_stem(name: str) -> str:
    return Path(str(name).replace("\\", "/")).name.lower()


def classify_plant_table(name: str, columns: Optional[list[str]] = None) -> Optional[str]:
    """Guess production / downtime / quality from a filename and optional headers."""
    stem = _zip_stem(name)
    for kind, hints in _KIND_NAME_HINTS.items():
        if any(h in stem for h in hints):
            return kind
    if columns:
        cols = {str(c).strip().lower() for c in columns}
        scores = {
            kind: sum(1 for h in hints if h in cols)
            for kind, hints in _KIND_COLUMN_HINTS.items()
        }
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
    return None


def _safe_zip_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    out: list[zipfile.ZipInfo] = []
    for info in zf.infolist():
        raw = info.filename.replace("\\", "/")
        if info.is_dir() or raw.endswith("/"):
            continue
        parts = Path(raw).parts
        if any(p in {".", ".."} or p.startswith("/") for p in parts):
            raise ValueError(f"Unsafe zip path: {info.filename}")
        base = Path(raw).name
        if not base or base.startswith(".") or "__macosx" in raw.lower():
            continue
        if not any(base.lower().endswith(suf) for suf in TABULAR_SUFFIXES):
            continue
        out.append(info)
    return out


def _named_bytes(name: str, raw: bytes) -> io.BytesIO:
    buf = io.BytesIO(raw)
    buf.name = name
    return buf


def load_zip_tables(zip_src) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """
    Load a plant ZIP with production / downtime / quality extracts.

    Members are classified by filename (production_logs.csv, sap_pm_downtime.csv, …)
    then by column names if the filename is generic.
    """
    source = zip_src
    if isinstance(zip_src, (str, Path)):
        source = Path(zip_src)
    try:
        zf = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise ValueError("Not a valid ZIP archive.") from exc

    tables: dict[str, pd.DataFrame] = {}
    log: list[dict[str, Any]] = []
    with zf:
        members = _safe_zip_members(zf)
        if not members:
            raise ValueError("ZIP has no CSV / Excel / JSON / Parquet files.")
        pending: list[tuple[str, pd.DataFrame]] = []
        for info in members:
            raw = zf.read(info)
            name = Path(info.filename.replace("\\", "/")).name
            df = load_tabular_file(_named_bytes(name, raw))
            kind = classify_plant_table(info.filename, list(df.columns))
            pending.append((kind or "", df))
            log.append(
                {
                    "member": info.filename,
                    "rows": int(len(df)),
                    "cols": int(df.shape[1]),
                    "kind": kind or "unclassified",
                }
            )
        unclassified = [(k, df) for k, df in pending if not k]
        classified = [(k, df) for k, df in pending if k]
        leftover_kinds = [k for k in TABLE_KINDS if k not in {x[0] for x in classified}]
        for kind, df in classified:
            if kind in tables:
                # Keep the larger extract if two files map to the same table.
                if len(df) > len(tables[kind]):
                    tables[kind] = df
            else:
                tables[kind] = df
        for df, kind in zip((x[1] for x in unclassified), leftover_kinds):
            tables[kind] = df
            for item in log:
                if item["kind"] == "unclassified" and item["rows"] == len(df):
                    item["kind"] = f"{kind} (by order)"
                    break
    if not tables:
        raise ValueError("Could not classify any plant tables inside the ZIP.")
    return tables, log


def extract_zip_member_path(
    zip_src,
    dest_dir: Path,
    *,
    table_kind: Optional[str] = None,
) -> dict[str, Path]:
    """Extract classified plant tables from a ZIP onto disk (for DuckDB SQL slices)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tables, _log = load_zip_tables(zip_src)
    if table_kind:
        if table_kind not in tables:
            raise ValueError(
                f"ZIP has no {table_kind} table. Found: {', '.join(tables) or 'none'}."
            )
        tables = {table_kind: tables[table_kind]}
    out: dict[str, Path] = {}
    for kind, df in tables.items():
        path = dest_dir / f"{kind}.csv"
        df.to_csv(path, index=False)
        out[kind] = path
    return out


def suggest_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Intersect column names as candidate join keys."""
    common = sorted(set(left.columns) & set(right.columns))
    preferred_names = {
        "machine_id",
        "line_id",
        "shift",
        "shift_date",
        "timestamp",
        "date",
        "asset_id",
        "id",
    }
    preferred = [c for c in common if c.lower() in preferred_names]
    rest = [c for c in common if c not in preferred]
    return preferred + rest


def join_two(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "inner",
    on: Optional[list[str]] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    suffixes: tuple[str, str] = ("_l", "_r"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    how = (how or "inner").lower()
    if how not in JOIN_TYPES:
        raise ValueError(f"Unsupported join type: {how}. Use one of {list(JOIN_TYPES)}")

    meta: dict[str, Any] = {
        "how": how,
        "left_rows": len(left),
        "right_rows": len(right),
    }
    if on:
        merged = pd.merge(left, right, how=how, on=on, suffixes=suffixes)
        meta["keys"] = on
    elif left_on and right_on:
        merged = pd.merge(
            left, right, how=how, left_on=left_on, right_on=right_on, suffixes=suffixes
        )
        meta["keys"] = [left_on, right_on]
    else:
        keys = suggest_join_keys(left, right)
        if not keys:
            raise ValueError("No common columns to join on. Pick left_on/right_on explicitly.")
        merged = pd.merge(left, right, how=how, on=keys[:1], suffixes=suffixes)
        meta["keys"] = keys[:1]
        meta["auto_key"] = True

    meta["result_rows"] = len(merged)
    meta["result_cols"] = list(merged.columns)
    return merged, meta


def join_many(
    tables: dict[str, pd.DataFrame],
    steps: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Chain joins across 3+ named tables.

    steps example:
      [
        {"left": "production", "right": "downtime", "how": "left", "on": ["machine_id", "shift_date"]},
        {"left": "_result", "right": "quality", "how": "left", "on": ["machine_id", "shift_date"]},
      ]
    """
    if not tables:
        raise ValueError("No tables provided")
    if not steps:
        raise ValueError("Provide at least one join step")

    working = tables[steps[0]["left"]].copy()
    registry = dict(tables)
    logs: list[dict[str, Any]] = []

    for i, step in enumerate(steps):
        right_name = step["right"]
        if right_name not in registry:
            raise KeyError(f"Unknown right table: {right_name}")
        left_df = working if i > 0 or step.get("left") == "_result" else registry[step["left"]]
        right_df = registry[right_name]
        how = step.get("how", "inner")
        on = step.get("on")
        left_on = step.get("left_on")
        right_on = step.get("right_on")
        working, meta = join_two(
            left_df,
            right_df,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
        )
        meta["step"] = i + 1
        meta["left_name"] = step.get("left", "_result")
        meta["right_name"] = right_name
        logs.append(meta)
        registry["_result"] = working

    return working, logs


def plant_default_join(
    production: pd.DataFrame,
    downtime: pd.DataFrame,
    quality: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Opinionated 3-table plant join: production ← downtime ← quality."""
    tables = {"production": production, "downtime": downtime, "quality": quality}
    keys = suggest_join_keys(production, downtime)
    if not keys:
        keys = ["machine_id"]
    # Prefer compound keys when available
    preferred = [k for k in ["machine_id", "line_id", "shift_date", "shift"] if k in keys]
    on = preferred if preferred else keys[:2] if len(keys) >= 2 else keys[:1]

    steps = [
        {"left": "production", "right": "downtime", "how": "left", "on": on},
        {"left": "_result", "right": "quality", "how": "left", "on": on},
    ]
    return join_many(tables, steps)


def try_duckdb_join(
    production: pd.DataFrame,
    downtime: pd.DataFrame,
    quality: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Optional DuckDB SQL join for plant tables."""
    try:
        import duckdb
    except ImportError:
        return None

    con = duckdb.connect()
    con.register("production", production)
    con.register("downtime", downtime)
    con.register("quality", quality)
    sql = """
    SELECT
      p.*,
      d.downtime_minutes,
      d.downtime_code,
      d.downtime_category,
      d.event_id,
      q.good_count,
      q.reject_count,
      q.scrap_rate
    FROM production p
    LEFT JOIN downtime d
      ON p.machine_id = d.machine_id
     AND CAST(p.shift_date AS DATE) = CAST(d.shift_date AS DATE)
     AND p.shift = d.shift
    LEFT JOIN quality q
      ON p.machine_id = q.machine_id
     AND CAST(p.shift_date AS DATE) = CAST(q.shift_date AS DATE)
     AND p.shift = q.shift
    """
    try:
        return con.execute(sql).df()
    except Exception:
        return None
