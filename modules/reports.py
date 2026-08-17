"""HTML / PDF / email brief generation for OEE Pulse."""

from __future__ import annotations

import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional

from modules.config_secrets import demo_email_mode, get_email_from, get_email_to, get_smtp_config

ROOT = Path(__file__).resolve().parents[1]
DEMO_EMAIL_DIR = ROOT / "reports" / "output" / "email_demo"


def _rows(obj: Any, n: int = 8) -> list[dict[str, Any]]:
    if obj is None:
        return []
    if hasattr(obj, "to_dict") and hasattr(obj, "head"):
        try:
            if getattr(obj, "empty", False):
                return []
            return list(obj.head(n).to_dict("records"))
        except Exception:
            return []
    if isinstance(obj, list):
        return [r for r in obj[:n] if isinstance(r, dict)]
    return []


def _esc(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_html_brief(
    plant_name: str,
    oee_plant: dict[str, Any],
    insights: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    reliability: dict[str, Any],
    qa_answer: str = "",
    maintenance: Optional[dict[str, Any]] = None,
    quality: Optional[dict[str, Any]] = None,
) -> str:
    insight_html = "".join(
        f"<li><strong>[{i.get('priority','').upper()}]</strong> {i.get('message','')}</li>"
        for i in insights[:8]
    )
    pareto_html = "".join(
        f"<tr><td>{r.get('cause')}</td><td>{r.get('downtime_minutes', 0):.0f}</td>"
        f"<td>{float(r.get('pct', 0))*100:.1f}%</td></tr>"
        for r in pareto_rows[:8]
    )
    mttr = reliability.get("mttr_min", "—")
    mtbf = reliability.get("mtbf_min", "—")
    qa_block = ""
    if qa_answer:
        safe = (
            str(qa_answer)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        qa_block = f"""
  <div class="card">
    <h2>Manager Q&amp;A</h2>
    <p>{safe}</p>
  </div>"""

    maint_block = ""
    if maintenance and maintenance.get("ok"):
        hours = maintenance.get("hours_lost") or {}
        pu = maintenance.get("planned_vs_unplanned") or {}
        inspect_rows = _rows(maintenance.get("inspect_this_week"), 5)
        inspect_html = "".join(
            f"<tr><td>{_esc(r.get('machine_id'))}</td><td>{_esc(r.get('line_id'))}</td>"
            f"<td>{float(r.get('remaining_risk', 0)):.0f}</td>"
            f"<td>{_esc(r.get('why', ''))}</td></tr>"
            for r in inspect_rows
        )
        maint_block = f"""
  <div class="card">
    <h2>Maintenance</h2>
    <p>{_esc(hours.get('narrative') or '')}</p>
    <p>Planned {float(pu.get('planned_min') or 0):.0f} min
       ({float(pu.get('planned_pct') or 0)*100:.0f}%) vs
       unplanned {float(pu.get('unplanned_min') or 0):.0f} min
       ({float(pu.get('unplanned_pct') or 0)*100:.0f}%).</p>
    <table class="pareto">
      <thead><tr><th>Inspect this week</th><th>Line</th><th>Risk</th><th>Why</th></tr></thead>
      <tbody>{inspect_html or '<tr><td colspan="4">No ranking</td></tr>'}</tbody>
    </table>
  </div>"""

    quality_block = ""
    if quality and quality.get("ok"):
        k = quality.get("kpis") or {}
        cards = quality.get("cards") or []
        killer = cards[0].get("message", "") if cards else ""
        def_rows = _rows(quality.get("defect_pareto"), 6)
        def_html = "".join(
            f"<tr><td>{_esc(r.get('defect_code'))}</td>"
            f"<td>{float(r.get('rejects', 0)):.0f}</td>"
            f"<td>{float(r.get('pct', 0))*100:.1f}%</td></tr>"
            for r in def_rows
        )
        cpk = quality.get("cpk") or {}
        cpk_line = ""
        if cpk.get("ok"):
            cpk_line = f"Cp={cpk.get('cp')} · Cpk={cpk.get('cpk')} on {cpk.get('measurement')}."
        quality_block = f"""
  <div class="card">
    <h2>Quality</h2>
    <p>Scrap {float(k.get('scrap_rate') or 0)*100:.1f}% ·
       FPY {float(k.get('fpy') or 0)*100:.1f}% ·
       Quality OEE {float(k.get('quality_oee') or 0)*100:.1f}% ·
       Rejects {float(k.get('rejects') or 0):.0f}. {cpk_line}</p>
    <p>{_esc(killer)}</p>
    <table class="pareto">
      <thead><tr><th>Defect</th><th>Rejects</th><th>%</th></tr></thead>
      <tbody>{def_html or '<tr><td colspan="3">No defect Pareto</td></tr>'}</tbody>
    </table>
  </div>"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>OEE Pulse Brief — {plant_name}</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; color: #1e293b; margin: 24px; background: #f8fafc; }}
  .card {{ background: #fff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 20px; margin-bottom: 16px; }}
  h1 {{ color: #0f172a; margin: 0 0 4px; }}
  .accent {{ color: #d97706; }}
  .metrics td {{ padding: 8px 16px 8px 0; font-size: 18px; }}
  table.pareto {{ border-collapse: collapse; width: 100%; }}
  table.pareto th, table.pareto td {{ border-bottom: 1px solid #e2e8f0; text-align: left; padding: 8px; }}
  .footer {{ color: #64748b; font-size: 12px; margin-top: 24px; }}
</style></head><body>
  <div class="card">
    <h1>OEE Pulse <span class="accent">Plant Brief</span></h1>
    <div>{plant_name} · {now} UTC</div>
  </div>
  <div class="card">
    <h2>OEE Cockpit</h2>
    <table class="metrics">
      <tr>
        <td><strong>OEE</strong><br>{float(oee_plant.get('oee',0))*100:.1f}%</td>
        <td><strong>Availability</strong><br>{float(oee_plant.get('availability',0))*100:.1f}%</td>
        <td><strong>Performance</strong><br>{float(oee_plant.get('performance',0))*100:.1f}%</td>
        <td><strong>Quality</strong><br>{float(oee_plant.get('quality',0))*100:.1f}%</td>
      </tr>
    </table>
    <p>MTTR: {mttr} min · MTBF: {mtbf} min</p>
  </div>
  <div class="card">
    <h2>Manager Insights</h2>
    <ul>{insight_html or '<li>No insights generated.</li>'}</ul>
  </div>
  <div class="card">
    <h2>Downtime Pareto</h2>
    <table class="pareto">
      <thead><tr><th>Cause</th><th>Minutes</th><th>%</th></tr></thead>
      <tbody>{pareto_html or '<tr><td colspan="3">No downtime data</td></tr>'}</tbody>
    </table>
  </div>
  {qa_block}
  {maint_block}
  {quality_block}
  <div class="footer">Generated by OEE Pulse · Reduce unplanned downtime · Protect the schedule</div>
</body></html>"""


def write_html_report(html: str, out_dir: Path | str = "reports/output") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"oee_pulse_brief_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html"
    path.write_text(html, encoding="utf-8")
    return path


def _pdf_safe(text: str) -> str:
    """Map common Unicode to Latin-1-safe ASCII for Helvetica core fonts."""
    repl = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u00d7": "x",
        "\u2248": "~",
        "\u2192": "->",
    }
    out = str(text)
    for a, b in repl.items():
        out = out.replace(a, b)
    return out.encode("latin-1", errors="replace").decode("latin-1")


def write_pdf_report(html_summary: dict[str, Any], out_dir: Path | str = "reports/output") -> Path:
    """Lightweight PDF brief via fpdf2 (no browser dependency)."""
    from fpdf import FPDF

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"oee_pulse_brief_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "OEE Pulse Plant Brief", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(
        0,
        8,
        _pdf_safe(
            f"{html_summary.get('plant_name', 'Plant')}  |  "
            f"{datetime.now(timezone.utc).isoformat()}"
        ),
        ln=True,
    )
    pdf.ln(4)
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "OEE Metrics", ln=True)
    pdf.set_font("Helvetica", "", 12)
    oee = html_summary.get("oee_plant", {})
    for label, key in [
        ("OEE", "oee"),
        ("Availability", "availability"),
        ("Performance", "performance"),
        ("Quality", "quality"),
    ]:
        pdf.cell(0, 7, f"{label}: {float(oee.get(key, 0))*100:.1f}%", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Insights", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for i in html_summary.get("insights", [])[:6]:
        msg = _pdf_safe(str(i.get("message", ""))[:220])
        pdf.multi_cell(0, 6, _pdf_safe(f"- [{i.get('priority','')}] {msg}"))
        pdf.ln(1)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Top Downtime", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for r in html_summary.get("pareto_rows", [])[:6]:
        pdf.cell(
            0,
            6,
            _pdf_safe(
                f"{r.get('cause')}: {float(r.get('downtime_minutes', 0)):.0f} min "
                f"({float(r.get('pct', 0))*100:.1f}%)"
            ),
            ln=True,
        )
    maint = html_summary.get("maintenance") or {}
    if maint.get("ok"):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Maintenance", ln=True)
        pdf.set_font("Helvetica", "", 11)
        hours = maint.get("hours_lost") or {}
        if hours.get("narrative"):
            pdf.multi_cell(0, 6, _pdf_safe(str(hours["narrative"])[:500]))
            pdf.ln(1)
        for r in _rows(maint.get("inspect_this_week"), 5):
            pdf.cell(
                0,
                6,
                _pdf_safe(
                    f"Inspect {r.get('machine_id')} (line {r.get('line_id')}): "
                    f"risk {float(r.get('remaining_risk', 0)):.0f}"
                ),
                ln=True,
            )
    qual = html_summary.get("quality") or {}
    if qual.get("ok"):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Quality", ln=True)
        pdf.set_font("Helvetica", "", 11)
        k = qual.get("kpis") or {}
        pdf.cell(
            0,
            6,
            _pdf_safe(
                f"Scrap {float(k.get('scrap_rate') or 0)*100:.1f}%  |  "
                f"FPY {float(k.get('fpy') or 0)*100:.1f}%  |  "
                f"Quality OEE {float(k.get('quality_oee') or 0)*100:.1f}%"
            ),
            ln=True,
        )
        cards = qual.get("cards") or []
        if cards:
            pdf.multi_cell(0, 6, _pdf_safe(str(cards[0].get("message", ""))[:400]))
        for r in _rows(qual.get("defect_pareto"), 5):
            pdf.cell(
                0,
                6,
                _pdf_safe(
                    f"{r.get('defect_code')}: {float(r.get('rejects', 0)):.0f} rejects "
                    f"({float(r.get('pct', 0))*100:.1f}%)"
                ),
                ln=True,
            )
    pdf.output(str(path))
    return path


def send_email_brief(
    html: str,
    subject: Optional[str] = None,
    to_addr: Optional[str] = None,
    demo_mode: Optional[bool] = None,
) -> dict[str, Any]:
    """Send HTML brief via SMTP, or save to disk in demo mode when SMTP is missing."""
    if demo_mode is None:
        demo_mode = demo_email_mode()

    cfg = get_smtp_config()
    to_addr = (to_addr or cfg["to_addr"] or get_email_to()).strip()
    from_addr = cfg["from_addr"] or get_email_from()
    subject = subject or f"OEE Pulse Daily Brief — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

    if demo_mode or not cfg.get("host"):
        DEMO_EMAIL_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = DEMO_EMAIL_DIR / f"email_to_{to_addr.replace('@', '_at_')}_{stamp}.html"
        path.write_text(html, encoding="utf-8")
        meta = DEMO_EMAIL_DIR / f"email_to_{to_addr.replace('@', '_at_')}_{stamp}.json"
        meta.write_text(
            f'{{"to": "{to_addr}", "from": "{from_addr}", "subject": "{subject}", "saved": "{path.name}"}}',
            encoding="utf-8",
        )
        return {
            "ok": True,
            "demo": True,
            "message": (
                f"Demo / no-SMTP mode: brief saved for {to_addr} at {path}. "
                "Set SMTP_HOST + credentials and OEE_PULSE_DEMO_MODE=false to send for real."
            ),
            "subject": subject,
            "to": to_addr,
            "path": str(path),
        }

    host = cfg["host"]
    port = int(cfg["port"] or 587)
    user = cfg["user"]
    password = cfg["password"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            if cfg.get("use_tls", True):
                server.starttls()
            if user and password:
                server.login(user, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
    except Exception as exc:
        # Fall back to disk so Cloud demos never hard-fail
        DEMO_EMAIL_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = DEMO_EMAIL_DIR / f"email_fallback_{stamp}.html"
        path.write_text(html, encoding="utf-8")
        return {
            "ok": False,
            "demo": True,
            "error": str(exc),
            "message": f"SMTP failed ({exc}); saved fallback to {path}",
            "to": to_addr,
            "path": str(path),
        }

    return {"ok": True, "demo": False, "to": to_addr, "subject": subject, "message": f"Sent to {to_addr}"}


AUTOMATION_NOTE = """
**Email automation MVP (Cloud-safe)**

- **Now:** use **Email report now** to send (or demo-save) the latest HTML brief to your configured recipient (`EMAIL_TO` / secrets).
- **Recipient:** saved in session prefs + sidebar; also read from Streamlit secrets / `.env`.
- **Schedule (external):** Streamlit Community Cloud does not run long-lived background schedulers.
  Use one of:
  1. GitHub Actions / cron hitting a webhook (future), or
  2. Local cron: `0 6 * * * curl ...` / script that builds the brief and SMTP-sends, or
  3. Open the app daily and click **Email report now**.

**Future connector:** inbound email → auto-report (IMAP poll of CSV attachments) is planned —
not enabled on Cloud yet to keep the MVP stable and secret-safe.
""".strip()
