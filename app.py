"""OEE Pulse — Industry 4.0 OEE + downtime analytics SaaS MVP."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from modules.config_secrets import demo_email_mode, get_email_to, llm_status
from modules.data_integration import (
    JOIN_TYPES,
    join_many,
    load_tabular_file,
    plant_default_join,
    suggest_join_keys,
    try_duckdb_join,
)
from modules.downtime_analysis import downtime_pareto, mttr_mtbf, top_chronic_machines
from modules.insights_engine import ask_oee_question, generate_insights
from modules.maintenance_analysis import analyze_maintenance, frame_ok as maint_frame_ok
from modules.oee_engine import oee_summary, prepare_oee_frame
from modules.optuna_tuner import tune_and_forecast
from modules.quality_analysis import analyze_quality, frame_ok as quality_frame_ok
from modules.quality_checks import build_quality_report, clean_plant_frame
from modules.reports import (
    AUTOMATION_NOTE,
    build_html_brief,
    send_email_brief,
    write_html_report,
    write_pdf_report,
)
from modules.sample_data import generate_sample_plant
from modules import session_store
from ui.session import (
    append_chat,
    ensure_session_id,
    get_active_frame,
    init_session,
    load_persisted_session,
    persist_current_session,
    reset_data,
)
from ui.theme import apply_theme, hero

load_dotenv()
ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_data"

st.set_page_config(
    page_title="OEE Pulse",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session()
apply_theme()

NAV = [
    "Upload & Integrate",
    "Clean & Quality",
    "OEE Cockpit",
    "Downtime Analysis",
    "Maintenance",
    "Quality",
    "Insights",
    "Reports",
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(color="#e2e8f0", family="IBM Plex Sans"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def _style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#475569")
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#475569")
    return fig


def ensure_sample_csvs() -> None:
    prod = SAMPLE_DIR / "production_logs.csv"
    if not prod.exists():
        generate_sample_plant(out_dir=SAMPLE_DIR)


def load_sample_into_session() -> None:
    ensure_sample_csvs()
    st.session_state.production_df = pd.read_csv(SAMPLE_DIR / "production_logs.csv")
    st.session_state.downtime_df = pd.read_csv(SAMPLE_DIR / "downtime_events.csv")
    st.session_state.quality_df = pd.read_csv(SAMPLE_DIR / "quality_rejects.csv")
    for col in ("shift_date", "start_time", "end_time"):
        for key in ("production_df", "downtime_df", "quality_df"):
            df = st.session_state[key]
            if df is not None and col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    _, logs = plant_default_join(
        st.session_state.production_df,
        st.session_state.downtime_df,
        st.session_state.quality_df,
    )
    # Prefer shift-level production+quality (downtime minutes already on production sample)
    prod = st.session_state.production_df.merge(
        st.session_state.quality_df,
        on=["shift_date", "shift", "line_id", "machine_id"],
        how="left",
        suffixes=("", "_q"),
    )
    st.session_state.integrated_df = prod
    st.session_state.cleaned_df = None
    st.session_state.oee_summary = None
    st.session_state.insights = None
    st.session_state.join_logs = logs
    st.session_state.sample_loaded = True
    ensure_session_id()
    st.session_state.session_title = "Sample plant"
    persist_current_session(title="Sample plant")


def _render_chat_qa(frame: pd.DataFrame, key_prefix: str = "qa") -> None:
    st.markdown("#### Ask the plant data")
    st.caption(
        "Manager Q&A grounded in OEE / downtime metrics. "
        f"LLM: {llm_status()}"
    )
    hist = st.session_state.get("chat_history") or []
    for msg in hist[-8:]:
        role = msg.get("role", "user")
        src = msg.get("source") or ""
        label = "You" if role == "user" else f"OEE Pulse ({src or 'assistant'})"
        st.markdown(f"**{label}:** {msg.get('content', '')}")

    question = st.text_input(
        "Question",
        placeholder="e.g. Which line lost the most availability to changeovers?",
        key=f"{key_prefix}_question",
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        ask = st.button("Ask", type="primary", key=f"{key_prefix}_ask", use_container_width=True)
    with c2:
        if st.button("Clear chat", key=f"{key_prefix}_clear"):
            st.session_state.chat_history = []
            persist_current_session()
            st.rerun()

    if ask and question.strip():
        append_chat("user", question.strip())
        result = ask_oee_question(
            question.strip(),
            frame,
            downtime=st.session_state.downtime_df,
            quality=st.session_state.quality_df,
            insights=st.session_state.insights,
            history=st.session_state.chat_history,
        )
        append_chat("assistant", result.get("answer", ""), source=result.get("source", ""))
        persist_current_session()
        st.rerun()


def _render_cards(cards: list) -> None:
    for item in cards or []:
        pr = item.get("priority", "medium")
        st.markdown(
            f'<div class="priority-{pr}"><strong>{item.get("title")}</strong><br>{item.get("message")}</div>',
            unsafe_allow_html=True,
        )


def _store_dept_analyses(maint=None, qual=None) -> None:
    if maint is not None:
        st.session_state.maintenance_analysis = maint
    if qual is not None:
        st.session_state.quality_analysis = qual


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### OEE Pulse")
    st.caption(os.getenv("PLANT_NAME", "North Plant"))
    page = st.radio("Navigate", NAV, label_visibility="collapsed")
    st.divider()
    if st.button("Load sample plant data", use_container_width=True):
        load_sample_into_session()
        st.success("Sample production, downtime & quality loaded.")
    if st.button("Save session now", use_container_width=True):
        sid = persist_current_session()
        st.success(f"Saved {sid}")
    if st.button("Reset session", use_container_width=True):
        reset_data()
        st.info("Session cleared (new empty session id).")
    st.divider()
    st.markdown("**Recent sessions**")
    recent = session_store.list_sessions(12)
    if not recent:
        st.caption("No saved sessions yet.")
    else:
        labels = {
            r["id"]: f"{r['title']} · {r['updated_at'][:16]}"
            for r in recent
        }
        choice = st.selectbox(
            "Reload",
            options=list(labels.keys()),
            format_func=lambda i: labels.get(i, i),
            key="recent_session_pick",
        )
        if st.button("Load selected", use_container_width=True):
            if load_persisted_session(choice):
                st.success(f"Loaded {choice}")
                st.rerun()
            else:
                st.error("Could not load session.")
    st.divider()
    email_default = st.session_state.get("email_to") or get_email_to()
    st.session_state.email_to = st.text_input("Report email", value=email_default)
    st.caption(
        "Demo email: "
        + ("on (disk save)" if demo_email_mode() else "off (SMTP)")
    )
    cur = st.session_state.get("session_id")
    if cur:
        st.caption(f"Session: {st.session_state.get('session_title') or cur}")


hero()


# ── Upload & Integrate ────────────────────────────────────────────────────────
if page == "Upload & Integrate":
    st.subheader("Upload & Integrate")
    st.write(
        "Upload production logs, downtime events, and quality/rejects — or load synthetic plant data."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        up_prod = st.file_uploader("Production logs", type=["csv", "xlsx", "tsv", "json"], key="up_prod")
    with c2:
        up_dt = st.file_uploader("Downtime codes / events", type=["csv", "xlsx", "tsv", "json"], key="up_dt")
    with c3:
        up_q = st.file_uploader("Rejects / quality", type=["csv", "xlsx", "tsv", "json"], key="up_q")

    if up_prod:
        st.session_state.production_df = load_tabular_file(up_prod)
    if up_dt:
        st.session_state.downtime_df = load_tabular_file(up_dt)
    if up_q:
        st.session_state.quality_df = load_tabular_file(up_q)

    tabs = st.tabs(["Production", "Downtime", "Quality", "Join"])
    with tabs[0]:
        if st.session_state.production_df is not None:
            st.dataframe(st.session_state.production_df.head(100), use_container_width=True)
            st.caption(f"{len(st.session_state.production_df)} rows")
        else:
            st.info("No production data yet.")
    with tabs[1]:
        if st.session_state.downtime_df is not None:
            st.dataframe(st.session_state.downtime_df.head(100), use_container_width=True)
            st.caption(f"{len(st.session_state.downtime_df)} rows")
        else:
            st.info("No downtime data yet.")
    with tabs[2]:
        if st.session_state.quality_df is not None:
            st.dataframe(st.session_state.quality_df.head(100), use_container_width=True)
        else:
            st.info("No quality data yet.")
    with tabs[3]:
        ready = all(
            st.session_state[k] is not None
            for k in ("production_df", "downtime_df", "quality_df")
        )
        if not ready:
            st.warning("Load all three tables (or sample data) before joining.")
        else:
            how = st.selectbox("Join type", list(JOIN_TYPES.keys()), index=1)
            st.caption(JOIN_TYPES[how])
            keys = suggest_join_keys(st.session_state.production_df, st.session_state.downtime_df)
            default_keys = keys[: min(3, len(keys))] if keys else []
            on = st.multiselect("Join keys", keys, default=default_keys)
            engine = st.radio("Engine", ["pandas", "duckdb (optional)"], horizontal=True)
            if st.button("Run 3-table integration", type="primary"):
                if engine.startswith("duckdb"):
                    duck = try_duckdb_join(
                        st.session_state.production_df,
                        st.session_state.downtime_df,
                        st.session_state.quality_df,
                    )
                    if duck is not None:
                        st.session_state.integrated_df = duck
                        st.session_state.join_logs = [{"engine": "duckdb", "rows": len(duck)}]
                    else:
                        st.warning("DuckDB join failed — falling back to pandas.")
                        engine = "pandas"
                if (not engine.startswith("duckdb")) or (st.session_state.integrated_df is None):
                    steps = [
                        {
                            "left": "production",
                            "right": "downtime",
                            "how": how,
                            "on": on if on else None,
                        },
                        {
                            "left": "_result",
                            "right": "quality",
                            "how": how,
                            "on": on if on else None,
                        },
                    ]
                    join_on = on if on else suggest_join_keys(
                        st.session_state.production_df, st.session_state.quality_df
                    )[:3]
                    prod_q = st.session_state.production_df.merge(
                        st.session_state.quality_df,
                        on=join_on,
                        how="left",
                        suffixes=("", "_q"),
                    )
                    dt = st.session_state.downtime_df
                    minutes_col = "downtime_minutes" if "downtime_minutes" in dt.columns else None
                    if minutes_col and all(c in dt.columns for c in join_on):
                        agg = dt.groupby(list(join_on), as_index=False)[minutes_col].sum()
                        if "downtime_minutes" in prod_q.columns:
                            prod_q = prod_q.drop(columns=["downtime_minutes"])
                        prod_q = prod_q.merge(agg, on=join_on, how="left")
                        prod_q["downtime_minutes"] = prod_q["downtime_minutes"].fillna(0)
                    st.session_state.integrated_df = prod_q
                    _, logs = join_many(
                        {
                            "production": st.session_state.production_df,
                            "downtime": st.session_state.downtime_df,
                            "quality": st.session_state.quality_df,
                        },
                        steps,
                    )
                    st.session_state.join_logs = logs
                st.session_state.cleaned_df = None
                persist_current_session(title=st.session_state.get("session_title") or "Integrated data")
                st.success(f"Integrated frame: {st.session_state.integrated_df.shape}")
            if st.session_state.join_logs is not None:
                st.json(st.session_state.join_logs)
            if st.session_state.integrated_df is not None:
                st.dataframe(st.session_state.integrated_df.head(50), use_container_width=True)


# ── Clean & Quality ───────────────────────────────────────────────────────────
elif page == "Clean & Quality":
    st.subheader("Clean & Quality")
    src = st.session_state.integrated_df
    if src is None and st.session_state.production_df is not None:
        src = st.session_state.production_df
    if src is None:
        st.warning("Load or integrate data first.")
    else:
        if st.button("Run industrial clean + quality checks", type="primary"):
            cleaned, clog = clean_plant_frame(src)
            if st.session_state.downtime_df is not None:
                st.session_state.downtime_df, _ = clean_plant_frame(st.session_state.downtime_df)
            if st.session_state.quality_df is not None:
                st.session_state.quality_df, _ = clean_plant_frame(st.session_state.quality_df)
            st.session_state.cleaned_df = cleaned
            st.session_state.quality_report = build_quality_report(cleaned)
            st.session_state.clean_log = clog
            st.session_state.oee_summary = None
            persist_current_session(title=st.session_state.get("session_title") or "Cleaned data")

        if st.session_state.cleaned_df is not None:
            st.write("**Cleaning log**")
            st.json(st.session_state.get("clean_log") or {})
            report = st.session_state.quality_report
            if report is None:
                report = {}
            summary = report.get("summary", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Checks", summary.get("total", 0))
            m2.metric("Pass", summary.get("pass", 0))
            m3.metric("Warn", summary.get("warn", 0))
            m4.metric("Fail", summary.get("fail", 0))
            checks_df = pd.DataFrame(report.get("checks", []))
            if not checks_df.empty:
                st.dataframe(checks_df, use_container_width=True)
            st.dataframe(st.session_state.cleaned_df.head(80), use_container_width=True)
        else:
            st.info("Run cleaning to populate the quality report.")


# ── OEE Cockpit ───────────────────────────────────────────────────────────────
elif page == "OEE Cockpit":
    st.subheader("OEE Cockpit")
    frame = get_active_frame()
    if frame is None:
        st.warning("Integrate (and optionally clean) data first — or load sample plant data.")
    else:
        summary = oee_summary(frame)
        st.session_state.oee_summary = summary
        plant = summary["plant"]
        a, b, c, d, e = st.columns(5)
        a.metric("OEE", f"{float(plant.get('oee', 0))*100:.1f}%")
        b.metric("Availability", f"{float(plant.get('availability', 0))*100:.1f}%")
        c.metric("Performance", f"{float(plant.get('performance', 0))*100:.1f}%")
        d.metric("Quality", f"{float(plant.get('quality', 0))*100:.1f}%")
        e.metric("Gap to 85%", f"{summary['world_class_gap']*100:.1f} pts")

        work = summary["frame"]
        loss_df = pd.DataFrame(
            {
                "component": ["Availability loss", "Performance loss", "Quality loss", "OEE realized"],
                "value": [
                    float(work["loss_availability"].mean()),
                    float(work["loss_performance"].mean()),
                    float(work["loss_quality"].mean()),
                    float(work["oee"].mean()),
                ],
            }
        )
        fig_loss = px.bar(
            loss_df,
            x="component",
            y="value",
            title="Average loss decomposition",
            color="component",
            color_discrete_sequence=["#64748b", "#94a3b8", "#cbd5e1", "#f59e0b"],
        )
        st.plotly_chart(_style_fig(fig_loss), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            by_line = summary["by_line"]
            if isinstance(by_line, pd.DataFrame) and not by_line.empty:
                fig = px.bar(
                    by_line,
                    x="line_id",
                    y=["availability", "performance", "quality", "oee"],
                    barmode="group",
                    title="OEE by line",
                    color_discrete_sequence=["#64748b", "#94a3b8", "#e2e8f0", "#f59e0b"],
                )
                st.plotly_chart(_style_fig(fig), use_container_width=True)
                st.dataframe(by_line, use_container_width=True)
        with c2:
            by_machine = summary["by_machine"]
            if isinstance(by_machine, pd.DataFrame) and not by_machine.empty:
                fig = px.bar(
                    by_machine.sort_values("oee"),
                    x="machine_id",
                    y="oee",
                    title="OEE by machine",
                    color="oee",
                    color_continuous_scale=["#334155", "#f59e0b"],
                )
                st.plotly_chart(_style_fig(fig), use_container_width=True)

        by_shift = summary["by_shift"]
        if isinstance(by_shift, pd.DataFrame) and not by_shift.empty:
            fig = go.Figure()
            for col, color in [
                ("availability", "#94a3b8"),
                ("performance", "#64748b"),
                ("quality", "#e2e8f0"),
                ("oee", "#f59e0b"),
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=by_shift["shift"],
                        y=by_shift[col],
                        mode="lines+markers",
                        name=col.title(),
                        line=dict(color=color, width=3),
                    )
                )
            fig.update_layout(title="Shift breakdown", yaxis_tickformat=".0%")
            st.plotly_chart(_style_fig(fig), use_container_width=True)

        with st.expander("Row-level OEE detail"):
            st.dataframe(prepare_oee_frame(frame).head(200), use_container_width=True)


# ── Downtime Analysis ─────────────────────────────────────────────────────────
elif page == "Downtime Analysis":
    st.subheader("Downtime Analysis")
    dt = st.session_state.downtime_df
    if dt is None:
        alt = get_active_frame()
        if alt is not None and any(
            c in alt.columns for c in ("downtime_code", "downtime_category", "downtime_minutes")
        ):
            dt = alt
            st.info("Using active production frame for downtime metrics (no separate events table).")
        else:
            st.warning("Load downtime events (or sample data).")
            dt = None

    if dt is not None:
        try:
            pareto = downtime_pareto(dt)
            if pareto.empty:
                st.warning("No downtime minutes to chart.")
            else:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=pareto["cause"],
                        y=pareto["downtime_minutes"],
                        name="Minutes",
                        marker_color="#64748b",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pareto["cause"],
                        y=pareto["cum_pct"],
                        name="Cumulative %",
                        yaxis="y2",
                        mode="lines+markers",
                        line=dict(color="#f59e0b", width=3),
                    )
                )
                fig.update_layout(
                    title="Downtime Pareto",
                    yaxis=dict(title="Minutes"),
                    yaxis2=dict(
                        title="Cumulative %",
                        overlaying="y",
                        side="right",
                        tickformat=".0%",
                        range=[0, 1.05],
                    ),
                )
                st.plotly_chart(_style_fig(fig), use_container_width=True)
                st.dataframe(pareto, use_container_width=True)
        except Exception as exc:
            st.error(f"Pareto failed: {exc}")

        rel = mttr_mtbf(dt)
        c1, c2, c3, c4 = st.columns(4)
        if rel.get("ok"):
            c1.metric("Events", rel.get("events", 0))
            c2.metric("MTTR (min)", rel.get("mttr_min", "—"))
            c3.metric("MTBF (min)", rel.get("mtbf_min", "—"))
            c4.metric("Total downtime (min)", rel.get("total_downtime_min", "—"))
        else:
            st.info(rel.get("reason", "Reliability metrics unavailable"))

        chronic = top_chronic_machines(dt)
        if chronic is not None and not chronic.empty:
            st.write("**Chronic downtime machines**")
            fig = px.bar(
                chronic,
                x=chronic.columns[0],
                y="downtime_minutes",
                title="Top machines by downtime",
                color="downtime_minutes",
                color_continuous_scale=["#334155", "#f59e0b"],
            )
            st.plotly_chart(_style_fig(fig), use_container_width=True)
            st.dataframe(chronic, use_container_width=True)

        if "line_id" in dt.columns and "downtime_minutes" in dt.columns:
            by_line = dt.groupby("line_id")["downtime_minutes"].sum().reset_index()
            fig = px.pie(
                by_line,
                names="line_id",
                values="downtime_minutes",
                title="Downtime share by line",
                color_discrete_sequence=["#334155", "#64748b", "#f59e0b", "#94a3b8"],
            )
            st.plotly_chart(_style_fig(fig), use_container_width=True)


# ── Maintenance ───────────────────────────────────────────────────────────────
elif page == "Maintenance":
    st.subheader("Maintenance")
    st.caption(
        "Reliability overlay on the same plant data — asset risk, planned vs unplanned, "
        "and inspect-this-week ranking tied to OEE Availability hours lost."
    )
    dt = st.session_state.downtime_df
    if not maint_frame_ok(dt):
        st.warning("Downtime events table missing. Upload events or click **Load sample plant data**.")
        alt = get_active_frame()
        if maint_frame_ok(alt) and any(
            c in alt.columns for c in ("downtime_minutes", "downtime_code", "downtime_category")
        ):
            dt = alt
            st.info("Using the active production frame as a downtime overlay.")
        else:
            dt = None

    if maint_frame_ok(dt):
        maint = analyze_maintenance(
            dt,
            production=st.session_state.production_df,
            oee_frame=get_active_frame(),
        )
        _store_dept_analyses(maint=maint)

        if not maint.get("ok"):
            st.warning(maint.get("reason", "Maintenance analysis unavailable."))
        else:
            rel = maint.get("reliability") or {}
            hours = maint.get("hours_lost") or {}
            pu = maint.get("planned_vs_unplanned") or {}
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Events", rel.get("events", "—"))
            c2.metric("MTTR (min)", rel.get("mttr_min", "—"))
            c3.metric("MTBF (min)", rel.get("mtbf_min", "—"))
            c4.metric("Hours lost", f"{float(hours.get('downtime_hours') or 0):.1f}")
            c5.metric("Unplanned", f"{float(pu.get('unplanned_pct') or 0)*100:.0f}%")
            if hours.get("narrative"):
                st.info(hours["narrative"])
            _render_cards(maint.get("cards") or [])

            inspect = maint.get("inspect_this_week")
            if isinstance(inspect, pd.DataFrame) and not inspect.empty:
                st.write("**Inspect this week**")
                fig = px.bar(
                    inspect.sort_values("remaining_risk"),
                    x="remaining_risk",
                    y="machine_id",
                    color="line_id",
                    orientation="h",
                    title="Remaining-risk ranking (rule-based + optional sensor IsolationForest)",
                    color_discrete_sequence=["#64748b", "#94a3b8", "#f59e0b"],
                    hover_data=["mttr_min", "failure_freq_per_day", "unplanned_share", "why"],
                )
                st.plotly_chart(_style_fig(fig), use_container_width=True)
                iso = maint.get("iso") or {}
                if iso.get("ok"):
                    st.caption(
                        f"IsolationForest on sensors {iso.get('features')} · "
                        f"{iso.get('anomaly_count')} anomalous rows / {iso.get('rows')}"
                    )
                else:
                    st.caption(f"Sensor IsolationForest skipped: {iso.get('reason', 'no numeric sensors')}")
                st.dataframe(inspect, use_container_width=True)

            assets = maint.get("asset_risk")
            if isinstance(assets, pd.DataFrame) and not assets.empty:
                st.write("**Asset / line risk (MTTR, MTBF, frequency, longest event)**")
                st.dataframe(assets, use_container_width=True)

            scope = st.radio("Reason Pareto scope", ["machine_id", "line_id"], horizontal=True)
            rpareto = maint.get("reason_by_machine") if scope == "machine_id" else maint.get("reason_by_line")
            if isinstance(rpareto, pd.DataFrame) and not rpareto.empty:
                fig = px.bar(
                    rpareto,
                    x="cause",
                    y="downtime_minutes",
                    color="asset",
                    barmode="group",
                    title=f"Downtime reasons by {scope.replace('_id', '')}",
                    color_discrete_sequence=["#334155", "#64748b", "#94a3b8", "#f59e0b", "#e2e8f0"],
                )
                st.plotly_chart(_style_fig(fig), use_container_width=True)

            by_line = pu.get("by_line")
            if isinstance(by_line, pd.DataFrame) and not by_line.empty:
                fig = px.bar(
                    by_line,
                    x="line_id",
                    y="downtime_minutes",
                    color="event_class",
                    barmode="stack",
                    title=f"Planned vs unplanned by line ({pu.get('source', 'heuristic')})",
                    color_discrete_map={"Planned": "#94a3b8", "Unplanned": "#f59e0b"},
                )
                st.plotly_chart(_style_fig(fig), use_container_width=True)

            longest = maint.get("longest_events")
            if isinstance(longest, pd.DataFrame) and not longest.empty:
                st.write("**Longest downtime events**")
                st.dataframe(longest, use_container_width=True)

            frame = get_active_frame()
            if frame is not None:
                st.divider()
                _render_chat_qa(frame, key_prefix="maint")


# ── Quality ───────────────────────────────────────────────────────────────────
elif page == "Quality":
    st.subheader("Quality")
    st.caption(
        "SPC / scrap / defects overlay on the same plant data — FPY, defect Pareto, "
        "and contribution to the Quality component of OEE."
    )
    qdf = st.session_state.quality_df
    if not quality_frame_ok(qdf):
        st.warning("Quality / rejects table missing. Upload rejects or click **Load sample plant data**.")
        alt = get_active_frame()
        if quality_frame_ok(alt) and any(
            c in alt.columns for c in ("scrap_rate", "reject_count", "good_count", "defect_code")
        ):
            qdf = alt
            st.info("Using the active production frame as a quality overlay.")
        else:
            qdf = None

    if quality_frame_ok(qdf):
        qan = analyze_quality(qdf, oee_frame=get_active_frame(), production=st.session_state.production_df)
        _store_dept_analyses(qual=qan)

        if not qan.get("ok"):
            st.warning(qan.get("reason", "Quality analysis unavailable."))
        else:
            k = qan.get("kpis") or {}
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Scrap", f"{float(k.get('scrap_rate') or 0)*100:.1f}%")
            m2.metric("First-pass yield", f"{float(k.get('fpy') or 0)*100:.1f}%")
            m3.metric("Quality OEE", f"{float(k.get('quality_oee') or 0)*100:.1f}%")
            m4.metric("Rejects", f"{float(k.get('rejects') or 0):,.0f}")
            loss = k.get("quality_loss_oee_pts")
            m5.metric("Q-loss (OEE pts)", "—" if loss is None else f"{float(loss)*100:.1f}")
            if qan.get("defect_synthesized"):
                st.caption("Defect codes were synthesized — upload defect_code/type for a true Pareto.")
            _render_cards(qan.get("cards") or [])

            by_line = k.get("by_line")
            if isinstance(by_line, pd.DataFrame) and not by_line.empty:
                fig = px.bar(
                    by_line,
                    x="line_id",
                    y=["scrap_rate_weighted", "fpy_weighted"],
                    barmode="group",
                    title="Scrap vs first-pass yield by line",
                    color_discrete_sequence=["#f59e0b", "#94a3b8"],
                )
                st.plotly_chart(_style_fig(fig), use_container_width=True)

            pareto = qan.get("defect_pareto")
            if isinstance(pareto, pd.DataFrame) and not pareto.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=pareto["defect_code"],
                        y=pareto["rejects"],
                        name="Rejects",
                        marker_color="#64748b",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pareto["defect_code"],
                        y=pareto["cum_pct"],
                        name="Cumulative %",
                        yaxis="y2",
                        mode="lines+markers",
                        line=dict(color="#f59e0b", width=3),
                    )
                )
                fig.update_layout(
                    title="Defect Pareto",
                    yaxis=dict(title="Rejects"),
                    yaxis2=dict(
                        title="Cumulative %",
                        overlaying="y",
                        side="right",
                        tickformat=".0%",
                        range=[0, 1.05],
                    ),
                )
                st.plotly_chart(_style_fig(fig), use_container_width=True)
                st.dataframe(pareto, use_container_width=True)

            spc = qan.get("spc") or {}
            if spc.get("ok"):
                st.write(f"**SPC lite** — {spc.get('method')} on `{spc.get('metric')}`")
                plant = spc.get("plant") or {}
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("CL", plant.get("cl", "—"))
                s2.metric("UCL", plant.get("ucl", "—"))
                s3.metric("LCL", plant.get("lcl", "—"))
                s4.metric("Out of control", plant.get("ooc_count", 0))
                chart = spc.get("chart_df")
                if isinstance(chart, pd.DataFrame) and not chart.empty:
                    xcol = "shift_date" if "shift_date" in chart.columns else "point"
                    fig = px.line(
                        chart,
                        x=xcol,
                        y="xbar",
                        color="line_id" if "line_id" in chart.columns else None,
                        markers=True,
                        title=f"X-bar / MR — {spc.get('metric')} by shift/line",
                        color_discrete_sequence=["#94a3b8", "#f59e0b", "#e2e8f0"],
                    )
                    if plant.get("ucl") is not None:
                        fig.add_hline(y=plant["ucl"], line_dash="dash", line_color="#ef4444")
                        fig.add_hline(y=plant["cl"], line_dash="dot", line_color="#f59e0b")
                        fig.add_hline(y=plant["lcl"], line_dash="dash", line_color="#64748b")
                    st.plotly_chart(_style_fig(fig), use_container_width=True)
            elif spc.get("reason"):
                st.caption(f"SPC skipped: {spc['reason']}")

            cpk = qan.get("cpk") or {}
            if cpk.get("ok"):
                st.write("**Cp / Cpk lite**")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Cp", cpk.get("cp"))
                p2.metric("Cpk", cpk.get("cpk"))
                p3.metric("LSL / USL", f"{cpk.get('lsl')} / {cpk.get('usl')}")
                p4.metric("Mean", cpk.get("mean"))
                by_cpk = cpk.get("by_line")
                if isinstance(by_cpk, pd.DataFrame) and not by_cpk.empty:
                    fig = px.bar(
                        by_cpk,
                        x="line_id",
                        y=["cp", "cpk"],
                        barmode="group",
                        title=f"Capability by line ({cpk.get('measurement')})",
                        color_discrete_sequence=["#94a3b8", "#f59e0b"],
                    )
                    st.plotly_chart(_style_fig(fig), use_container_width=True)
                    st.dataframe(by_cpk, use_container_width=True)
            else:
                st.caption(cpk.get("reason") or "Cp/Cpk not available.")

            frame = get_active_frame()
            if frame is not None:
                st.divider()
                _render_chat_qa(frame, key_prefix="quality")


# ── Insights ──────────────────────────────────────────────────────────────────
elif page == "Insights":
    st.subheader("Insights")
    frame = get_active_frame()
    if frame is None:
        st.warning("Need integrated production data.")
    else:
        n_trials = st.slider("Optuna trials (forecast)", 5, 40, 15)
        use_optuna = st.checkbox("Enable Optuna hyperparameter tuning", value=True)
        if st.button("Generate manager insights", type="primary"):
            insights = generate_insights(
                frame, st.session_state.downtime_df, st.session_state.quality_df
            )
            st.session_state.insights = insights
            st.session_state.forecast = tune_and_forecast(
                frame, n_trials=n_trials, use_optuna=use_optuna
            )
            persist_current_session()

        insights = st.session_state.get("insights")
        if insights:
            for item in insights:
                pr = item.get("priority", "medium")
                st.markdown(
                    f'<div class="priority-{pr}"><strong>{item.get("title")}</strong><br>{item.get("message")}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Generate insights (or ask a question below — offline answers work immediately).")

        forecast = st.session_state.get("forecast")
        if forecast:
            st.divider()
            st.write("**Scrap / downtime rate forecast & anomalies**")
            if forecast.get("ok"):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Target", forecast.get("target"))
                m2.metric("Next forecast", f"{forecast.get('next_period_forecast', 0):.4f}")
                m3.metric("Recent avg", f"{forecast.get('recent_avg', 0):.4f}")
                m4.metric("Test MAE", f"{forecast.get('mae', 0):.5f}")
                st.write(
                    f"Anomalies: {forecast.get('anomaly_count')} "
                    f"({forecast.get('anomaly_rate_pct')}%) · "
                    f"Optuna: {forecast.get('optuna_used')} · "
                    f"Best params: {forecast.get('best_params', {})}"
                )
                if forecast.get("feature_importance"):
                    fi = pd.DataFrame(
                        list(forecast["feature_importance"].items()),
                        columns=["feature", "importance"],
                    )
                    fig = px.bar(
                        fi,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Feature importance",
                        color_discrete_sequence=["#f59e0b"],
                    )
                    st.plotly_chart(_style_fig(fig), use_container_width=True)
            else:
                st.info(forecast.get("reason", "Forecast unavailable"))

        st.divider()
        _render_chat_qa(frame, key_prefix="insights")


# ── Reports ───────────────────────────────────────────────────────────────────
elif page == "Reports":
    st.subheader("Reports & Email Brief")
    frame = get_active_frame()
    if frame is None:
        st.warning("Need data before reporting.")
    else:
        summary = st.session_state.get("oee_summary")
        if summary is None:
            summary = oee_summary(frame)
            st.session_state.oee_summary = summary

        insights = st.session_state.get("insights")
        if not insights:
            insights = generate_insights(
                frame, st.session_state.downtime_df, st.session_state.quality_df
            )
            st.session_state.insights = insights

        plant_name = os.getenv("PLANT_NAME", "North Plant")
        pareto_rows = []
        reliability = {}
        maint = None
        qual = None
        if maint_frame_ok(st.session_state.downtime_df):
            try:
                pareto_rows = downtime_pareto(st.session_state.downtime_df).to_dict("records")
            except Exception:
                pareto_rows = []
            reliability = mttr_mtbf(st.session_state.downtime_df)
            maint = analyze_maintenance(
                st.session_state.downtime_df,
                production=st.session_state.production_df,
                oee_frame=frame,
            )
            _store_dept_analyses(maint=maint)
        if quality_frame_ok(st.session_state.quality_df):
            qual = analyze_quality(
                st.session_state.quality_df,
                oee_frame=frame,
                production=st.session_state.production_df,
            )
            _store_dept_analyses(qual=qual)

        last_assistant = ""
        for msg in reversed(st.session_state.get("chat_history") or []):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break

        html = build_html_brief(
            plant_name=plant_name,
            oee_plant=summary["plant"],
            insights=insights,
            pareto_rows=pareto_rows,
            reliability=reliability,
            qa_answer=last_assistant[:1200],
            maintenance=maint,
            quality=qual,
        )
        st.session_state.last_report_html = html
        st.components.v1.html(html, height=520, scrolling=True)

        st.divider()
        _render_chat_qa(frame, key_prefix="reports")

        st.divider()
        st.markdown("#### Export & email")
        to_addr = st.session_state.get("email_to") or get_email_to()
        to_addr = st.text_input("Send to", value=to_addr, key="report_email_to")
        st.session_state.email_to = to_addr

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Export HTML", use_container_width=True):
                path = write_html_report(html, ROOT / "reports" / "output")
                persist_current_session()
                st.success(f"Wrote {path}")
                st.download_button("Download HTML", data=html, file_name=path.name, mime="text/html")
        with c2:
            if st.button("Export PDF", use_container_width=True):
                path = write_pdf_report(
                    {
                        "plant_name": plant_name,
                        "oee_plant": summary["plant"],
                        "insights": insights,
                        "pareto_rows": pareto_rows,
                        "maintenance": maint,
                        "quality": qual,
                    },
                    ROOT / "reports" / "output",
                )
                persist_current_session()
                st.success(f"Wrote {path}")
                st.download_button(
                    "Download PDF",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/pdf",
                )
        with c3:
            if st.button("Email report now", use_container_width=True, type="primary"):
                result = send_email_brief(html, to_addr=to_addr)
                persist_current_session()
                if result.get("ok"):
                    st.success(result.get("message") or f"Sent to {result.get('to')}")
                else:
                    st.error(result.get("error") or result.get("message") or "Send failed")
                st.json(result)

        with st.expander("Automation note (schedule / inbound email)"):
            st.markdown(AUTOMATION_NOTE)
            st.text_area(
                "Optional schedule reminder (saved with session)",
                value=st.session_state.get("schedule_note") or "Daily 06:00 — Email report now / external cron",
                key="schedule_note_input",
            )
            if st.button("Save schedule note"):
                st.session_state.schedule_note = st.session_state.get("schedule_note_input", "")
                persist_current_session()
                st.success("Saved.")
