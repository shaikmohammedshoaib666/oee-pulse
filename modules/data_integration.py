"""Multi-file plant data integration with SQL-style joins."""

from __future__ import annotations

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


def load_tabular_file(uploaded_file) -> pd.DataFrame:
    """Load csv/tsv/xlsx/json into a DataFrame from a Streamlit UploadedFile or path."""
    name = getattr(uploaded_file, "name", str(uploaded_file)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
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
