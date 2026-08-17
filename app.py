"""OEE Pulse — Industry 4.0 OEE + downtime analytics SaaS MVP."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from modules.data_integration import (
    JOIN_TYPES,
    join_many,
    load_tabular_file,
    plant_default_join,
    suggest_join_keys,
    try_duckdb_join,
)
from modules.downtime_analysis import downtime_pareto, mttr_mtbf, top_chronic_machines
from modules.insights_engine import generate_insights
from modules.oee_engine import aggregate_oee, oee_summary, prepare_oee_frame
from modules.optuna_tuner import tune_and_forecast
from modules.quality_checks import build_quality_report, clean_plant_frame
from modules.reports import build_html_brief, send_email_brief, write_html_report, write_pdf_report
from modules.sample_data import generate_sample_plant
from ui.session import init_session, reset_data
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
    for col in ("shift_date",):
        for key in ("production_df", "downtime_df", "quality_df"):
            df = st.session_state[key]
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    integrated, logs = plant_default_join(
        st.session_state.production_df,
        st.session_state.downtime_df,
        st.session_state.quality_df,
    )
    # Prefer shift-level production+quality with aggregated downtime
    prod = st.session_state.production_df.merge(
        st.session_state.quality_df,
        on=["shift_date", "shift", "line_id", "machine_id"],
        how="left",
        suffixes=("", "_q"),
    )
    st.session_state.integrated_df = prod
    st.session_state.join_logs = logs
    st.session_state.sample_loaded = True


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### OEE Pulse")
    st.caption(os.getenv("PLANT_NAME", "North Plant"))
    page = st.radio("Navigate", NAV, label_visibility="collapsed")
    st.divider()
    if st.button("Load sample plant data", use_container_width=True):
        load_sample_into_session()
        st.success("Sample production, downtime & quality loaded.")
    if st.button("Reset session", use_container_width=True):
        reset_data()
        st.info("Session cleared.")
    st.divider()
    st.caption("Demo email mode: " + os.getenv("OEE_PULSE_DEMO_MODE", "true"))


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
            on = st.multiselect("Join keys", keys, default=keys[: min(3, len(keys))] or keys)
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
                if not engine.startswith("duckdb") or st.session_state.integrated_df is None:
                    steps = [
                        {
                            "left": "production",
                            "right": "downtime",
                            "how": how,
                            "on": on or None,
                        },
                        {
                            "left": "_result",
                            "right": "quality",
                            "how": how,
                            "on": on or None,
                        },
                    ]
                    # Shift-level product for OEE: production ⋈ quality, keep event downtime separate
                    prod_q = st.session_state.production_df.merge(
                        st.session_state.quality_df,
                        on=on if on else suggest_join_keys(
                            st.session_state.production_df, st.session_state.quality_df
                        )[:3],
                        how="left",
                        suffixes=("", "_q"),
                    )
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
                st.success(f"Integrated frame: {st.session_state.integrated_df.shape}")
            if st.session_state.join_logs:
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
            # Also clean downtime / quality tables if present
            if st.session_state.downtime_df is not None:
                st.session_state.downtime_df, _ = clean_plant_frame(st.session_state.downtime_df)
            if st.session_state.quality_df is not None:
                st.session_state.quality_df, _ = clean_plant_frame(st.session_state.quality_df)
            st.session_state.cleaned_df = cleaned
            st.session_state.quality_report = build_quality_report(cleaned)
            st.session_state.clean_log = clog

        if st.session_state.cleaned_df is not None:
            st.write("**Cleaning log**")
            st.json(getattr(st.session_state, "clean_log", {}))
            report = st.session_state.quality_report or {}
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
    frame = st.session_state.cleaned_df
    if frame is None:
        frame = st.session_state.integrated_df
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
        # Waterfall-style loss stacked bar
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
        st.warning("Load downtime events (or sample data).")
    else:
        try:
            pareto = downtime_pareto(dt)
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
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", tickformat=".0%", range=[0, 1.05]),
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
        if not chronic.empty:
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

        if "line_id" in dt.columns:
            by_line = dt.groupby("line_id")["downtime_minutes"].sum().reset_index()
            fig = px.pie(
                by_line,
                names="line_id",
                values="downtime_minutes",
                title="Downtime share by line",
                color_discrete_sequence=["#334155", "#64748b", "#f59e0b", "#94a3b8"],
            )
            st.plotly_chart(_style_fig(fig), use_container_width=True)


# ── Insights ──────────────────────────────────────────────────────────────────
elif page == "Insights":
    st.subheader("Insights")
    frame = st.session_state.cleaned_df or st.session_state.integrated_df
    if frame is None:
        st.warning("Need integrated production data.")
    else:
        n_trials = st.slider("Optuna trials (forecast)", 5, 40, 15)
        use_optuna = st.checkbox("Enable Optuna hyperparameter tuning", value=True)
        if st.button("Generate manager insights", type="primary"):
            insights = generate_insights(frame, st.session_state.downtime_df)
            st.session_state.insights = insights
            st.session_state.forecast = tune_and_forecast(frame, n_trials=n_trials, use_optuna=use_optuna)

        if st.session_state.insights:
            for item in st.session_state.insights:
                pr = item.get("priority", "medium")
                st.markdown(
                    f'<div class="priority-{pr}"><strong>{item.get("title")}</strong><br>{item.get("message")}</div>',
                    unsafe_allow_html=True,
                )

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


# ── Reports ───────────────────────────────────────────────────────────────────
elif page == "Reports":
    st.subheader("Reports & Email Brief")
    frame = st.session_state.cleaned_df or st.session_state.integrated_df
    if frame is None:
        st.warning("Need data before reporting.")
    else:
        summary = st.session_state.oee_summary or oee_summary(frame)
        insights = st.session_state.insights or generate_insights(frame, st.session_state.downtime_df)
        plant_name = os.getenv("PLANT_NAME", "North Plant")
        pareto_rows = []
        reliability = {}
        if st.session_state.downtime_df is not None:
            try:
                pareto_rows = downtime_pareto(st.session_state.downtime_df).to_dict("records")
            except Exception:
                pareto_rows = []
            reliability = mttr_mtbf(st.session_state.downtime_df)

        html = build_html_brief(
            plant_name=plant_name,
            oee_plant=summary["plant"],
            insights=insights,
            pareto_rows=pareto_rows,
            reliability=reliability,
        )
        st.components.v1.html(html, height=520, scrolling=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Export HTML", use_container_width=True):
                path = write_html_report(html, ROOT / "reports" / "output")
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
                    },
                    ROOT / "reports" / "output",
                )
                st.success(f"Wrote {path}")
                st.download_button(
                    "Download PDF",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/pdf",
                )
        with c3:
            if st.button("Send email brief", use_container_width=True, type="primary"):
                result = send_email_brief(html)
                if result.get("ok"):
                    st.success(result.get("message") or f"Sent to {result.get('to')}")
                else:
                    st.error(result.get("error", "Send failed"))
                st.json(result)
