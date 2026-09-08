"""URL / Google Drive ingest via DuckDB — stream large CSVs and SQL-slice before pandas."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

def _http():
    import requests

    return requests

USER_AGENT = (
    "Mozilla/5.0 (compatible; OEEPulse/1.0; "
    "+https://github.com/shaikmohammedshoaib666/oee-pulse)"
)
REQUEST_HEADERS = {"User-Agent": USER_AGENT}

_GDRIVE_ID_PATTERNS = (
    re.compile(r"drive\.google\.com/file/d/([^/?#]+)"),
    re.compile(r"drive\.google\.com/open\?[^#]*\bid=([^&#]+)"),
    re.compile(r"drive\.google\.com/uc\?(?:export=download&)?[^#]*\bid=([^&#]+)"),
    re.compile(r"drive\.usercontent\.google\.com/download\?[^#]*\bid=([^&#]+)"),
    re.compile(r"docs\.google\.com/spreadsheets/d/([^/?#]+)"),
)
_KAGGLE_PAGE = re.compile(r"kaggle\.com/(?:datasets|competitions)/([^/?#]+/[^/?#]+)")
_KAGGLE_SCHEME = re.compile(r"^kaggle://([^/]+/[^/]+)(?:/(.+))?$", re.I)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TIME_COLS = {
    "timestamp",
    "start_time",
    "end_time",
    "datetime",
    "shift_date",
    "date",
    "event_time",
}
_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|PRAGMA|EXPORT|IMPORT|INSTALL|LOAD)\b",
    re.IGNORECASE,
)

TABLE_KINDS = ("production", "downtime", "quality")

# Templates use {{source}} so .format() leaves `{source}` for the loader to fill.
INGEST_SQL_PRESETS: dict[str, dict[str, Any]] = {
    "first_n_rows": {
        "label": "Top N rows",
        "description": "First N rows in file order — quick head of a multi-GB extract.",
        "params": [("n", "Row limit", 50000, "int")],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "LIMIT {n}"
        ),
    },
    "last_n_rows": {
        "label": "Bottom N rows",
        "description": "Last N rows in file order (typical for newest records at the end).",
        "params": [("n", "Row limit", 50000, "int")],
        "template": (
            "SELECT * EXCLUDE (_oee_rn, _oee_n)\n"
            "FROM (\n"
            "  SELECT *,\n"
            "         row_number() OVER () AS _oee_rn,\n"
            "         count(*) OVER () AS _oee_n\n"
            "  FROM read_csv_auto('{{source}}', header=true)\n"
            ")\n"
            "WHERE _oee_rn > (_oee_n - {n})"
        ),
    },
    "middle_n_rows": {
        "label": "Middle N rows",
        "description": "Centered window around the middle of the file.",
        "params": [("n", "Window size", 50000, "int")],
        "template": (
            "SELECT * EXCLUDE (_oee_rn, _oee_n)\n"
            "FROM (\n"
            "  SELECT *,\n"
            "         row_number() OVER () AS _oee_rn,\n"
            "         count(*) OVER () AS _oee_n\n"
            "  FROM read_csv_auto('{{source}}', header=true)\n"
            ")\n"
            "WHERE _oee_rn BETWEEN CAST((_oee_n - {n}) / 2 AS BIGINT)\n"
            "                 AND CAST((_oee_n + {n}) / 2 AS BIGINT)"
        ),
    },
    "between_row_numbers": {
        "label": "Between row numbers",
        "description": "Inclusive start/end row positions (1-based, file order).",
        "params": [
            ("start_row", "Start row (1-based)", 1, "int"),
            ("end_row", "End row", 100000, "int"),
        ],
        "template": (
            "SELECT * EXCLUDE (_oee_rn)\n"
            "FROM (\n"
            "  SELECT *, row_number() OVER () AS _oee_rn\n"
            "  FROM read_csv_auto('{{source}}', header=true)\n"
            ")\n"
            "WHERE _oee_rn BETWEEN {start_row} AND {end_row}"
        ),
    },
    "between_ids": {
        "label": "Between IDs",
        "description": "Inclusive range on an id column (event_id, machine_id, lot, …).",
        "params": [
            ("id_col", "ID column", "event_id", "ident"),
            ("start_id", "Start ID", "1000", "str"),
            ("end_id", "End ID", "2000", "str"),
        ],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE (\n"
            "  try_cast('{start_id}' AS DOUBLE) IS NOT NULL\n"
            "  AND try_cast({id_col} AS DOUBLE) BETWEEN try_cast('{start_id}' AS DOUBLE)\n"
            "                                      AND try_cast('{end_id}' AS DOUBLE)\n"
            ") OR (\n"
            "  try_cast('{start_id}' AS DOUBLE) IS NULL\n"
            "  AND CAST({id_col} AS VARCHAR) BETWEEN '{start_id}' AND '{end_id}'\n"
            ")"
        ),
    },
    "date_range": {
        "label": "Between dates",
        "description": "Inclusive start, exclusive end on a timestamp / shift_date column.",
        "params": [
            ("start_date", "Start (YYYY-MM-DD)", "2025-07-01", "str"),
            ("end_date", "End (YYYY-MM-DD)", "2025-08-01", "str"),
            ("ts_col", "Timestamp column", "shift_date", "ident"),
        ],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE try_cast({ts_col} AS TIMESTAMP) >= TIMESTAMP '{start_date}'\n"
            "  AND try_cast({ts_col} AS TIMESTAMP) < TIMESTAMP '{end_date}'\n"
            "ORDER BY try_cast({ts_col} AS TIMESTAMP)"
        ),
    },
    "last_n_days": {
        "label": "Last N days",
        "description": "Time window ending at now (requires a timestamp/date column).",
        "params": [
            ("n", "Days back", 30, "int"),
            ("ts_col", "Timestamp column", "shift_date", "ident"),
        ],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE try_cast({ts_col} AS TIMESTAMP) >= current_timestamp - INTERVAL '{n} days'\n"
            "ORDER BY try_cast({ts_col} AS TIMESTAMP) DESC\n"
            "LIMIT 500000"
        ),
    },
    "filter_machine_id": {
        "label": "Filter machine_id",
        "description": "One asset from a plant-wide production / sensor extract.",
        "params": [
            ("machine_id", "Machine ID", "M101", "str"),
            ("n", "Row limit (0 = no cap)", 100000, "int"),
        ],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE CAST(machine_id AS VARCHAR) = '{machine_id}'\n"
            "{limit_clause}"
        ),
    },
    "filter_line_id": {
        "label": "Filter line_id",
        "description": "One line from a multi-line plant extract.",
        "params": [
            ("line_id", "Line ID", "L1", "str"),
            ("n", "Row limit (0 = no cap)", 100000, "int"),
        ],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE CAST(line_id AS VARCHAR) = '{line_id}'\n"
            "{limit_clause}"
        ),
    },
    "filter_shift": {
        "label": "Filter shift",
        "description": "Keep one shift (A / B / C) across the extract.",
        "params": [
            ("shift", "Shift", "A", "str"),
            ("n", "Row limit (0 = no cap)", 100000, "int"),
        ],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "WHERE CAST(shift AS VARCHAR) = '{shift}'\n"
            "{limit_clause}"
        ),
    },
    "sample_percent": {
        "label": "Random sample %",
        "description": "Explore a percentage of rows without loading the full file.",
        "params": [("sample_pct", "Sample percent (1–100)", 5, "float")],
        "template": (
            "SELECT *\n"
            "FROM read_csv_auto('{{source}}', header=true)\n"
            "USING SAMPLE {sample_pct}% (bernoulli)\n"
            "LIMIT 500000"
        ),
    },
}


def detect_source_kind(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return "empty"
    if _KAGGLE_SCHEME.match(u):
        return "kaggle_api"
    if "drive.google.com" in u or "docs.google.com/spreadsheets" in u or "drive.usercontent.google.com" in u:
        return "google_drive"
    if "kaggle.com" in u:
        return "kaggle_page"
    if u.startswith("file://"):
        return "local"
    parsed = Path(u)
    if parsed.exists() and parsed.is_file():
        return "local"
    return "https"


def extract_gdrive_file_id(url: str) -> Optional[str]:
    for pat in _GDRIVE_ID_PATTERNS:
        m = pat.search(url or "")
        if m:
            return m.group(1)
    return None


def extract_kaggle_slug(url: str) -> Optional[str]:
    m = _KAGGLE_PAGE.search(url or "")
    return m.group(1) if m else None


def _quote_ident(name: str) -> str:
    raw = str(name or "").strip()
    if not _IDENT.match(raw):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return raw


def _coerce_preset_param(value: Any, kind: str) -> Any:
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "ident":
        return _quote_ident(value)
    return str(value).replace("'", "''")


def list_ingest_presets() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for preset_id, spec in INGEST_SQL_PRESETS.items():
        out.append(
            {
                "id": preset_id,
                "label": spec["label"],
                "description": spec.get("description", ""),
                "params": list(spec.get("params") or []),
            }
        )
    return out


def build_preset_sql(preset_id: str, params: Optional[dict[str, Any]] = None) -> str:
    if preset_id not in INGEST_SQL_PRESETS:
        raise ValueError(f"Unknown ingest preset: {preset_id!r}")
    spec = INGEST_SQL_PRESETS[preset_id]
    merged: dict[str, Any] = {}
    n_limit = 0
    for name, _label, default, kind in spec.get("params") or []:
        raw = (params or {}).get(name, default)
        val = _coerce_preset_param(raw, kind)
        if name == "n":
            n_limit = int(val or 0)
            merged[name] = n_limit
        else:
            merged[name] = val

    limit_clause = f"LIMIT {n_limit}" if n_limit > 0 else ""
    template = spec["template"]
    if "{limit_clause}" in template:
        merged["limit_clause"] = limit_clause
    try:
        return template.format(**merged)
    except KeyError as exc:
        raise ValueError(f"Missing preset parameter for {preset_id}: {exc}") from exc


def default_ingest_sql(table_kind: str = "production") -> str:
    kind = (table_kind or "production").lower()
    if kind == "downtime":
        return (
            "SELECT *\n"
            "FROM read_csv_auto('{source}', header=true)\n"
            "WHERE try_cast(shift_date AS TIMESTAMP) >= TIMESTAMP '2025-07-01'\n"
            "  AND try_cast(shift_date AS TIMESTAMP) < TIMESTAMP '2025-08-01'\n"
            "ORDER BY try_cast(shift_date AS TIMESTAMP)\n"
            "LIMIT 100000"
        )
    if kind == "quality":
        return (
            "SELECT *\n"
            "FROM read_csv_auto('{source}', header=true)\n"
            "WHERE CAST(line_id AS VARCHAR) = 'L1'\n"
            "LIMIT 100000"
        )
    return (
        "SELECT *\n"
        "FROM read_csv_auto('{source}', header=true)\n"
        "WHERE 1 = 1  -- e.g. machine_id = 'M101' AND shift_date >= '2025-07-01'\n"
        "LIMIT 100000"
    )


def validate_ingest_sql(sql: str) -> str:
    text = (sql or "").strip()
    if not text:
        raise ValueError("SQL query is empty.")
    head = text.lstrip().split(None, 1)[0].upper()
    if head not in {"SELECT", "WITH"}:
        raise ValueError("Only SELECT (or WITH … SELECT) queries are allowed for ingest.")
    if ";" in text.rstrip().rstrip(";"):
        raise ValueError("Only one SQL statement allowed.")
    if _FORBIDDEN_SQL.search(text):
        raise ValueError("Only read-only SELECT queries are allowed.")
    if "{source}" not in text:
        raise ValueError("SQL must reference `{source}` (the resolved file path or URL).")
    return text


def duckdb_read_expr(path_or_url: str) -> str:
    escaped = str(path_or_url).replace("'", "''")
    low = path_or_url.lower().split("?")[0]
    if low.endswith(".parquet"):
        return f"read_parquet('{escaped}')"
    if low.endswith(".json"):
        return f"read_json_auto('{escaped}')"
    sep = "\\t" if low.endswith(".tsv") else ","
    return f"read_csv_auto('{escaped}', header=true, sep='{sep}')"


def _adapt_sql_for_path(sql: str, path_or_url: str) -> str:
    expr = duckdb_read_expr(path_or_url)
    adapted = re.sub(
        r"read_csv_auto\(\s*['\"]\{source\}['\"][^)]*\)",
        expr,
        sql,
        flags=re.IGNORECASE,
    )
    adapted = adapted.replace("{source}", str(path_or_url).replace("'", "''"))
    return adapted


def _duck_connect():
    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
    except Exception:
        pass
    return con


def _duckdb_read_sql(path_or_url: str, sql_template: str) -> pd.DataFrame:
    sql = validate_ingest_sql(sql_template)
    sql = _adapt_sql_for_path(sql, path_or_url)
    con = _duck_connect()
    return con.execute(sql).df()


def _duckdb_read(path_or_url: str, *, row_limit: Optional[int] = None) -> pd.DataFrame:
    limit_sql = f" LIMIT {int(row_limit)}" if row_limit and row_limit > 0 else ""
    sql = f"SELECT * FROM {duckdb_read_expr(path_or_url)}{limit_sql}"
    con = _duck_connect()
    return con.execute(sql).df()


def _coerce_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if str(col).lower() in _TIME_COLS:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def resolve_gdrive_download_url(file_id: str, session: Any = None) -> str:
    """Return a direct-download URL, handling Google's large-file confirm token."""
    sess = session or _http().Session()
    base = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = sess.get(base, stream=True, timeout=60, allow_redirects=True, headers=REQUEST_HEADERS)
    resp.raise_for_status()
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            return f"{base}&confirm={value}"
    ctype = (resp.headers.get("content-type") or "").lower()
    if "text/html" in ctype:
        m = re.search(r"confirm=([0-9A-Za-z_-]+)", resp.text or "")
        if m:
            return f"{base}&confirm={m.group(1)}"
        return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    return base


def _sheets_export_url(url: str, file_id: str) -> str:
    gid = parse_qs(urlparse(url).query).get("gid", [None])[0]
    export = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
    if gid:
        export += f"&gid={gid}"
    return export


def _kaggle_credentials() -> tuple[str, str]:
    user = (os.getenv("KAGGLE_USERNAME") or os.getenv("KAGGLE_USER") or "").strip()
    key = (os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN") or "").strip()
    if not user or not key:
        raise RuntimeError(
            "Kaggle ingest needs KAGGLE_USERNAME and KAGGLE_KEY in environment secrets "
            "(create at https://www.kaggle.com/settings → API)."
        )
    return user, key


def download_kaggle_file(owner_dataset: str, filename: Optional[str] = None) -> Path:
    user, key = _kaggle_credentials()
    os.environ.setdefault("KAGGLE_USERNAME", user)
    os.environ.setdefault("KAGGLE_KEY", key)
    owner, _, dataset = owner_dataset.partition("/")
    if not owner or not dataset:
        raise ValueError(f"Invalid Kaggle slug: {owner_dataset!r} (expected owner/dataset)")

    tmp_dir = Path(tempfile.mkdtemp(prefix="oee-kaggle-"))
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        if filename:
            api.dataset_download_file(owner, dataset, filename, path=str(tmp_dir), quiet=True)
            candidates = list(tmp_dir.glob(f"{Path(filename).stem}*"))
            if not candidates:
                candidates = list(tmp_dir.iterdir())
        else:
            api.dataset_download_files(f"{owner}/{dataset}", path=str(tmp_dir), quiet=True, unzip=True)
            candidates = [
                p
                for p in tmp_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".parquet", ".json", ".txt"}
            ]
            if not candidates:
                candidates = [p for p in tmp_dir.rglob("*") if p.is_file()]
        if not candidates:
            raise RuntimeError(f"No files downloaded from Kaggle dataset {owner}/{dataset}")
        candidates.sort(key=lambda p: (0 if p.suffix.lower() == ".csv" else 1, p.name))
        return candidates[0]
    except ImportError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Install kaggle for Kaggle links: pip install kaggle") from exc
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def resolve_source_to_fetch_url(url: str) -> tuple[str, dict[str, Any]]:
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Paste a URL first.")

    meta: dict[str, Any] = {"original_url": raw, "kind": detect_source_kind(raw)}

    if meta["kind"] == "local":
        path = Path(raw.replace("file://", "")).expanduser()
        if not path.exists() or not path.is_file():
            raise ValueError(f"Local file not found: {raw}")
        meta["local_path"] = str(path.resolve())
        meta["resolved_url"] = meta["local_path"]
        return meta["local_path"], meta

    m = _KAGGLE_SCHEME.match(raw)
    if m:
        slug, fname = m.group(1), m.group(2)
        meta["kaggle_slug"] = slug
        meta["kaggle_file"] = fname
        meta["local_path"] = str(download_kaggle_file(slug, fname))
        meta["resolved_url"] = meta["local_path"]
        return meta["local_path"], meta

    if meta["kind"] == "kaggle_page":
        slug = extract_kaggle_slug(raw)
        if not slug:
            raise ValueError("Could not parse Kaggle dataset slug from URL.")
        meta["kaggle_slug"] = slug
        meta["local_path"] = str(download_kaggle_file(slug))
        meta["resolved_url"] = meta["local_path"]
        return meta["local_path"], meta

    if meta["kind"] == "google_drive":
        file_id = extract_gdrive_file_id(raw)
        if not file_id:
            raise ValueError("Could not parse Google Drive file id from URL.")
        meta["gdrive_file_id"] = file_id
        if "docs.google.com/spreadsheets" in raw:
            resolved = _sheets_export_url(raw, file_id)
        else:
            resolved = resolve_gdrive_download_url(file_id)
        meta["resolved_url"] = resolved
        return resolved, meta

    if "dropbox.com" in raw and "dl=0" in raw:
        raw = raw.replace("dl=0", "dl=1")
    meta["resolved_url"] = raw
    return raw, meta


def _guess_cache_name(fetch_url: str, headers: dict[str, str], meta: dict[str, Any]) -> str:
    cd = headers.get("content-disposition") or headers.get("Content-Disposition") or ""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, re.I)
    if m:
        name = unquote(m.group(1).strip())
        if name:
            return Path(name).name
    if meta.get("gdrive_file_id"):
        return f"gdrive_{meta['gdrive_file_id']}.csv"
    parsed = urlparse(fetch_url)
    name = unquote(Path(parsed.path).name) or "remote_ingest.csv"
    if "." not in name:
        name += ".csv"
    return name


def _looks_like_html(path: Path, ctype: str) -> bool:
    if "text/html" in (ctype or "").lower():
        return True
    if path.stat().st_size > 4096:
        return False
    try:
        head = path.read_bytes()[:256].lstrip().lower()
    except OSError:
        return False
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def _cache_remote_file(fetch_url: str, dest_dir: Path, meta: dict[str, Any], *, chunk_mb: int = 8) -> Path:
    """Stream a remote file to disk so DuckDB can scan multi-GB sources without RAM blow-up."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_id = meta.get("gdrive_file_id")
    reuse = dest_dir / f"gdrive_{file_id}.csv" if file_id else None
    if reuse is not None and reuse.exists() and reuse.stat().st_size > 0:
        return reuse

    sess = _http().Session()
    urls_to_try = [fetch_url]
    if file_id:
        urls_to_try.extend(
            [
                f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
                f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            ]
        )

    last_err: Optional[Exception] = None
    for url in urls_to_try:
        try:
            with sess.get(url, stream=True, timeout=120, allow_redirects=True, headers=REQUEST_HEADERS) as resp:
                resp.raise_for_status()
                ctype = (resp.headers.get("content-type") or "").lower()
                name = _guess_cache_name(url, dict(resp.headers), meta)
                dest = dest_dir / name
                with dest.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            if _looks_like_html(dest, ctype):
                dest.unlink(missing_ok=True)
                last_err = RuntimeError(
                    "Google Drive returned HTML instead of the file. "
                    "Check sharing is 'Anyone with the link' (Viewer)."
                )
                continue
            return dest
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(str(last_err) if last_err else "Could not download remote file.")


def _pandas_excel(path: str, row_limit: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_excel(path)
    if row_limit and row_limit > 0:
        df = df.head(int(row_limit))
    return df


def _is_zip(path: str) -> bool:
    return path.lower().split("?")[0].endswith(".zip")


def _load_zip_resolved(
    path: str,
    *,
    table_kind: Optional[str],
    row_limit: Optional[int],
    sql_query: Optional[str],
    cache_dir: Path,
    meta: dict[str, Any],
    finish,
):
    from modules.data_integration import extract_zip_member_path, load_zip_tables

    tables, zlog = load_zip_tables(path)
    meta["zip_log"] = zlog
    extra = {k: v for k, v in tables.items()}
    chosen = table_kind if table_kind in extra else next(iter(extra))
    extracted = extract_zip_member_path(path, Path(cache_dir) / "unzipped", table_kind=None)
    meta["zip_paths"] = {k: str(p) for k, p in extracted.items()}
    meta["zip_tables"] = list(extra.keys())
    use_sql = bool((sql_query or "").strip())
    read_path = str(extracted.get(chosen) or next(iter(extracted.values())))
    extra.pop(chosen, None)
    if use_sql:
        df = _duckdb_read_sql(read_path, sql_query or "")
        engine = "zip-duckdb-sql"
    else:
        df = _duckdb_read(read_path, row_limit=row_limit)
        engine = "zip-duckdb"
    if row_limit and row_limit > 0:
        extra = {k: v.head(int(row_limit)).copy() for k, v in extra.items()}
    else:
        extra = {k: v.copy() for k, v in extra.items()}
    meta["extra_tables"] = extra
    return finish(df, engine)


def load_from_url(
    url: str,
    *,
    cache_dir: Path,
    row_limit: Optional[int] = None,
    force_cache: bool = False,
    sql_query: Optional[str] = None,
    table_kind: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a tabular dataset from HTTPS, Google Drive, Kaggle, local path, or a plant ZIP.

    Large Drive files (hundreds of MB to ~2 GB) are streamed to disk, then DuckDB
    SQL-slices so pandas only sees the requested rows.
    """
    fetch_target, meta = resolve_source_to_fetch_url(url)
    meta["row_limit"] = row_limit
    meta["sql_query"] = sql_query
    meta["table_kind"] = table_kind

    def _finish(df: pd.DataFrame, engine: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = _coerce_time_columns(df)
        drop = [c for c in ("_oee_rn", "_oee_n") if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
        extra = meta.get("extra_tables") or {}
        if extra:
            meta["extra_tables"] = {k: _coerce_time_columns(v) for k, v in extra.items()}
        meta["rows"] = len(df)
        meta["columns"] = list(df.columns)
        meta["engine"] = engine
        return df, meta

    use_sql = bool((sql_query or "").strip())
    cache_dir = Path(cache_dir)

    if meta.get("local_path"):
        local = str(Path(meta["local_path"]))
        if _is_zip(local):
            return _load_zip_resolved(
                local,
                table_kind=table_kind,
                row_limit=row_limit,
                sql_query=sql_query,
                cache_dir=cache_dir,
                meta=meta,
                finish=_finish,
            )
        low = local.lower()
        if low.endswith((".xlsx", ".xls", ".xlsm")):
            return _finish(_pandas_excel(local, row_limit=None if use_sql else row_limit), "pandas-excel")
        if use_sql:
            return _finish(_duckdb_read_sql(local, sql_query or ""), "duckdb-sql")
        return _finish(_duckdb_read(local, row_limit=row_limit), "duckdb")

    is_remote = fetch_target.startswith("http://") or fetch_target.startswith("https://")
    read_path = fetch_target

    if use_sql or force_cache or meta.get("kind") == "google_drive" or _is_zip(fetch_target):
        if is_remote:
            cached = _cache_remote_file(fetch_target, cache_dir, meta)
            meta["cached_path"] = str(cached)
            read_path = str(cached)
    elif is_remote:
        try:
            df = _duckdb_read(fetch_target, row_limit=row_limit)
            return _finish(df, "duckdb-httpfs")
        except Exception as stream_err:
            meta["stream_error"] = str(stream_err)
            cached = _cache_remote_file(fetch_target, cache_dir, meta)
            meta["cached_path"] = str(cached)
            read_path = str(cached)

    if _is_zip(read_path):
        return _load_zip_resolved(
            read_path,
            table_kind=table_kind,
            row_limit=row_limit,
            sql_query=sql_query,
            cache_dir=cache_dir,
            meta=meta,
            finish=_finish,
        )
    low = read_path.lower().split("?")[0]
    if low.endswith((".xlsx", ".xls", ".xlsm")):
        return _finish(_pandas_excel(read_path, row_limit=None if use_sql else row_limit), "pandas-excel")
    if use_sql:
        return _finish(_duckdb_read_sql(read_path, sql_query or ""), "duckdb-sql")
    return _finish(_duckdb_read(read_path, row_limit=row_limit), "duckdb")


def friendly_source_label(meta: dict[str, Any]) -> str:
    kind = meta.get("kind") or "url"
    if kind == "google_drive":
        return f"gdrive:{meta.get('gdrive_file_id', 'file')}"
    if kind == "local":
        return Path(meta.get("local_path") or meta.get("original_url") or "local").name
    if str(kind).startswith("kaggle"):
        return f"kaggle:{meta.get('kaggle_slug', 'dataset')}"
    return urlparse(meta.get("original_url", "url")).netloc or "url"
