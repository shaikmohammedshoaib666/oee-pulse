"""Session state helpers for OEE Pulse."""

from __future__ import annotations

from typing import Any

import streamlit as st


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
    "sample_loaded": False,
}


def init_session() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_data() -> None:
    for k in DEFAULTS:
        st.session_state[k] = DEFAULTS[k]
