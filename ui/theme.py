"""Industrial slate / steel / amber Streamlit theme for OEE Pulse."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.stApp {
  background: linear-gradient(165deg, #0f172a 0%, #1e293b 42%, #334155 100%);
  color: #e2e8f0;
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1220 0%, #1e293b 100%);
  border-right: 1px solid #475569;
}
section[data-testid="stSidebar"] * {
  color: #e2e8f0 !important;
}
h1, h2, h3 {
  color: #f8fafc !important;
  letter-spacing: -0.02em;
}
div[data-testid="stMetric"] {
  background: rgba(15, 23, 42, 0.72);
  border: 1px solid #475569;
  border-radius: 10px;
  padding: 12px 14px;
}
div[data-testid="stMetric"] label {
  color: #94a3b8 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #fbbf24 !important;
  font-family: 'IBM Plex Mono', monospace;
}
.op-hero {
  border: 1px solid #64748b;
  border-radius: 12px;
  padding: 18px 22px;
  background: linear-gradient(120deg, rgba(15,23,42,0.9), rgba(51,65,85,0.55));
  margin-bottom: 1rem;
}
.op-hero .brand {
  font-size: 1.65rem;
  font-weight: 700;
  color: #f8fafc;
}
.op-hero .brand span { color: #f59e0b; }
.op-hero .tag {
  color: #94a3b8;
  font-size: 0.95rem;
  margin-top: 4px;
}
.op-pill {
  display: inline-block;
  border: 1px solid #f59e0b;
  color: #fbbf24;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 0.75rem;
  margin-right: 6px;
}
.priority-high { border-left: 4px solid #ef4444; padding-left: 10px; margin: 8px 0; }
.priority-medium { border-left: 4px solid #f59e0b; padding-left: 10px; margin: 8px 0; }
.priority-low { border-left: 4px solid #22c55e; padding-left: 10px; margin: 8px 0; }
.stButton > button {
  background: #d97706;
  color: #0f172a;
  border: none;
  font-weight: 600;
}
.stButton > button:hover {
  background: #f59e0b;
  color: #0f172a;
}
hr { border-color: #475569; }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(subtitle: str = "Industry 4.0 OEE & downtime analytics for plant managers") -> None:
    st.markdown(
        f"""
        <div class="op-hero">
          <div><span class="op-pill">OEE</span><span class="op-pill">DOWNTIME</span><span class="op-pill">MAINT</span><span class="op-pill">QUALITY</span></div>
          <div class="brand">OEE <span>Pulse</span></div>
          <div class="tag">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
