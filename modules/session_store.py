"""Disk-backed session persistence (SQLite metadata + CSV frames).

Cloud-safe: writes under data/sessions/ in the app directory.
Closing the browser no longer wipes cleaned/integrated frames or chat.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = ROOT / "data" / "sessions"
DB_PATH = SESSIONS_DIR / "oee_pulse.db"

FRAME_KEYS = (
    "production_df",
    "downtime_df",
    "quality_df",
    "integrated_df",
    "cleaned_df",
)

META_KEYS = (
    "insights",
    "chat_history",
    "join_logs",
    "quality_report",
    "clean_log",
    "forecast",
    "email_to",
    "plant_name",
    "sample_loaded",
    "last_report_html",
    "schedule_note",
    "maintenance_analysis",
    "quality_analysis",
    "column_mappings",
    "finance_rates",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_store() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                plant_name TEXT,
                meta_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_prefs (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_store()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _session_dir(session_id: str) -> Path:
    path = SESSIONS_DIR / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "rows": len(obj)}
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def set_pref(key: str, value: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO app_prefs(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()


def get_pref(key: str, default: str = "") -> str:
    with connect() as conn:
        row = conn.execute("SELECT value FROM app_prefs WHERE key=?", (key,)).fetchone()
    return str(row["value"]) if row else default


def set_last_session_id(session_id: str) -> None:
    set_pref("last_session_id", session_id)


def get_last_session_id() -> Optional[str]:
    val = get_pref("last_session_id", "")
    return val or None


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    ensure_store()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at, plant_name FROM sessions "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    import shutil

    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        conn.commit()
    path = SESSIONS_DIR / session_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    if get_last_session_id() == session_id:
        set_pref("last_session_id", "")


def save_frames(session_id: str, frames: dict[str, Optional[pd.DataFrame]]) -> dict[str, str]:
    """Persist DataFrames as CSV under data/sessions/<id>/."""
    out_dir = _session_dir(session_id)
    saved: dict[str, str] = {}
    for key, df in frames.items():
        if df is None or not isinstance(df, pd.DataFrame):
            continue
        path = out_dir / f"{key}.csv"
        df.to_csv(path, index=False)
        saved[key] = str(path.relative_to(ROOT))
    return saved


def load_frames(session_id: str) -> dict[str, Optional[pd.DataFrame]]:
    out_dir = SESSIONS_DIR / session_id
    result: dict[str, Optional[pd.DataFrame]] = {k: None for k in FRAME_KEYS}
    if not out_dir.exists():
        return result
    for key in FRAME_KEYS:
        path = out_dir / f"{key}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                for col in df.columns:
                    cl = str(col).lower()
                    if any(k in cl for k in ("minute", "min", "count", "rate", "oee")):
                        continue
                    if (
                        any(x in cl for x in ("timestamp", "shift_date", "start_time", "end_time", "datetime"))
                        or cl.endswith("_date")
                        or cl.endswith("_time")
                        or cl in {"date", "time"}
                    ):
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                result[key] = df
            except Exception:
                result[key] = None
    return result


def save_session(
    session_id: str,
    *,
    title: Optional[str] = None,
    plant_name: str = "Plant",
    frames: Optional[dict[str, Optional[pd.DataFrame]]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> str:
    ensure_store()
    now = _utcnow()
    frames = frames or {}
    meta = meta or {}
    save_frames(session_id, frames)

    with connect() as conn:
        existing = conn.execute("SELECT created_at, title FROM sessions WHERE id=?", (session_id,)).fetchone()
        created = existing["created_at"] if existing else now
        final_title = title or (existing["title"] if existing else f"Session {session_id[:8]}")
        payload = json.dumps(_safe_json(meta), ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO sessions(id, title, created_at, updated_at, plant_name, meta_json)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              title=excluded.title,
              updated_at=excluded.updated_at,
              plant_name=excluded.plant_name,
              meta_json=excluded.meta_json
            """,
            (session_id, final_title, created, now, plant_name, payload),
        )
        conn.commit()
    set_last_session_id(session_id)
    return session_id


def load_session_meta(session_id: str) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute("SELECT meta_json, title, plant_name FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not row:
        return {}
    try:
        meta = json.loads(row["meta_json"] or "{}")
    except json.JSONDecodeError:
        meta = {}
    meta["_title"] = row["title"]
    meta["_plant_name"] = row["plant_name"]
    return meta


def session_exists(session_id: str) -> bool:
    with connect() as conn:
        row = conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
    return row is not None
