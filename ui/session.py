"""Session state helpers for OEE Pulse."""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
import streamlit as st

from modules import session_store

DEFAULTS: dict[str, Any] = {
    "production_df": None,
    "downtime_df": None,
    "quality_df": None,
    "integrated_df": None,
    "cleaned_df": None,
    "quality_report": None,
    "join_logs": None,
    "oee_summary": None,
    "insights": None,
    "forecast": None,
    "sample_loaded": False,
    "chat_history": [],
    "email_to": "",
    "session_id": None,
    "session_title": "",
    "last_report_html": None,
    "clean_log": None,
    "_session_hydrated": False,
}


def init_session() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v if not isinstance(v, list) else list(v)

    if not st.session_state.get("email_to"):
        try:
            from modules.config_secrets import get_email_to

            st.session_state.email_to = get_email_to()
        except Exception:
            st.session_state.email_to = os.getenv("EMAIL_TO", os.getenv("SMTP_TO", ""))

    # Reload last disk session once per browser session when memory is empty
    if not st.session_state.get("_session_hydrated"):
        st.session_state._session_hydrated = True
        if get_active_frame() is None:
            last_id = session_store.get_last_session_id()
            if last_id and session_store.session_exists(last_id):
                try:
                    load_persisted_session(last_id)
                except Exception:
                    pass


def reset_data(*, keep_email: bool = True) -> None:
    email = st.session_state.get("email_to", "")
    for k, v in DEFAULTS.items():
        st.session_state[k] = v if not isinstance(v, list) else list(v)
    st.session_state._session_hydrated = True
    if keep_email:
        st.session_state.email_to = email
    st.session_state.session_id = session_store.new_session_id()
    st.session_state.session_title = "New session"


def get_active_frame() -> Optional[pd.DataFrame]:
    """Prefer cleaned, then integrated, then production — never use DataFrame `or`."""
    cleaned = st.session_state.get("cleaned_df")
    if cleaned is not None:
        return cleaned
    integrated = st.session_state.get("integrated_df")
    if integrated is not None:
        return integrated
    production = st.session_state.get("production_df")
    if production is not None:
        return production
    return None


def ensure_session_id() -> str:
    sid = st.session_state.get("session_id")
    if not sid:
        sid = session_store.new_session_id()
        st.session_state.session_id = sid
        if not st.session_state.get("session_title"):
            st.session_state.session_title = f"Session {sid[:8]}"
    return sid


def collect_meta() -> dict[str, Any]:
    return {
        "insights": st.session_state.get("insights"),
        "chat_history": st.session_state.get("chat_history") or [],
        "join_logs": st.session_state.get("join_logs"),
        "quality_report": st.session_state.get("quality_report"),
        "clean_log": st.session_state.get("clean_log"),
        "forecast": st.session_state.get("forecast"),
        "email_to": st.session_state.get("email_to") or "",
        "plant_name": os.getenv("PLANT_NAME", "North Plant"),
        "sample_loaded": bool(st.session_state.get("sample_loaded")),
        "last_report_html": st.session_state.get("last_report_html"),
        "schedule_note": st.session_state.get("schedule_note"),
    }


def persist_current_session(*, title: Optional[str] = None) -> str:
    """Save frames + chat/insights to disk so browser close doesn't wipe work."""
    sid = ensure_session_id()
    frames = {k: st.session_state.get(k) for k in session_store.FRAME_KEYS}
    plant = os.getenv("PLANT_NAME", "North Plant")
    final_title = title or st.session_state.get("session_title") or f"Session {sid[:8]}"
    st.session_state.session_title = final_title
    session_store.save_session(
        sid,
        title=final_title,
        plant_name=plant,
        frames=frames,
        meta=collect_meta(),
    )
    return sid


def load_persisted_session(session_id: str) -> bool:
    if not session_store.session_exists(session_id):
        return False
    frames = session_store.load_frames(session_id)
    for k, df in frames.items():
        st.session_state[k] = df
    meta = session_store.load_session_meta(session_id)
    st.session_state.session_id = session_id
    st.session_state.session_title = meta.get("_title") or f"Session {session_id[:8]}"
    for key in (
        "insights",
        "chat_history",
        "join_logs",
        "quality_report",
        "clean_log",
        "forecast",
        "email_to",
        "sample_loaded",
        "last_report_html",
        "schedule_note",
    ):
        if key in meta and meta[key] is not None:
            st.session_state[key] = meta[key]
    if not st.session_state.get("chat_history"):
        st.session_state.chat_history = []
    st.session_state.oee_summary = None  # recompute on demand
    session_store.set_last_session_id(session_id)
    return True


def append_chat(role: str, content: str, source: str = "") -> None:
    hist = list(st.session_state.get("chat_history") or [])
    hist.append({"role": role, "content": content, "source": source})
    st.session_state.chat_history = hist[-40:]
