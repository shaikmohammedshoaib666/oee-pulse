"""One-click SAP-style PP / PM / QM extract templates (CSV + README)."""

from __future__ import annotations

import io
import zipfile
from typing import Any

import pandas as pd

README = """OEE Pulse — SAP-style extract templates
========================================

These CSVs use headers typical of SAP PP (production), PM (maintenance /
downtime), and QM (quality) extracts. They are not a live SAP connector —
export from SAP GUI / SQVI / CDS / your middleware into this shape, then
upload on **Upload & Integrate** and map columns (saved per plant).

Canonical fields OEE Pulse expects after mapping
------------------------------------------------
  line_id              Line / work center          (SAP: ARBPL)
  machine_id           Machine / asset             (SAP: EQUNR, TPLNR)
  shift                Shift                       (SAP: SCHICHT)
  shift_date           Shift / posting date        (SAP: BUDAT, QMDAT, ISDD)
  timestamp            Event timestamp
  planned_time_min     Planned minutes             (SAP: VGW02, BEZUG)
  run_time_min         Run minutes                 (SAP: ISTDZ)
  good_count           Good units                  (SAP: LMNGA)
  reject_count         Scrap / rejects             (SAP: XMNGA, FEHLERANZAHL)
  total_count          Total units                 (SAP: GAMNG)
  downtime_minutes     Downtime minutes            (SAP: AUSZT)
  downtime_code        Reason code                 (SAP: FECOD, URCOD)
  downtime_category    Reason text                 (SAP: QMTXT)
  start_time / end_time  Event window              (SAP: AUSVN, AUSBS)
  vibration_rms, temp_c, motor_current_a   Optional sensors for PdM risk

Files
-----
  sap_pp_production.csv   PP-ish confirmation / yield extract
  sap_pm_downtime.csv     PM-ish notification / breakdown extract
  sap_qm_quality.csv      QM-ish inspection / scrap extract

Join keys after mapping: shift_date + shift + line_id + machine_id.

Rows below are examples only — replace with your plant extract.
"""

PP_COLUMNS = [
    "WERKS",
    "ARBPL",
    "EQUNR",
    "BUDAT",
    "SCHICHT",
    "VGW02",
    "ISTDZ",
    "GAMNG",
    "LMNGA",
    "XMNGA",
    "MATNR",
    "AUFNR",
    "VIBRATION_RMS",
    "TEMP_C",
    "MOTOR_CURRENT_A",
]

PM_COLUMNS = [
    "WERKS",
    "EQUNR",
    "TPLNR",
    "ARBPL",
    "QMDAT",
    "SCHICHT",
    "AUSVN",
    "AUSBS",
    "AUSZT",
    "FECOD",
    "URCOD",
    "QMTXT",
    "AUFNR",
    "QMNUM",
]

QM_COLUMNS = [
    "WERKS",
    "PRUEFLOS",
    "MATNR",
    "ARBPL",
    "EQUNR",
    "BUDAT",
    "SCHICHT",
    "LMNGA",
    "FEHLERANZAHL",
    "FECOD",
    "MESSWERT",
    "TOLUNTER",
    "TOLOBER",
    "MERKNR",
]


def _pp_rows() -> list[dict[str, Any]]:
    return [
        {
            "WERKS": "1000",
            "ARBPL": "L1",
            "EQUNR": "M101",
            "BUDAT": "2025-07-01",
            "SCHICHT": "A",
            "VGW02": 480,
            "ISTDZ": 430,
            "GAMNG": 4200,
            "LMNGA": 4080,
            "XMNGA": 120,
            "MATNR": "SKU-A",
            "AUFNR": "10001234",
            "VIBRATION_RMS": 2.4,
            "TEMP_C": 47.1,
            "MOTOR_CURRENT_A": 12.2,
        },
        {
            "WERKS": "1000",
            "ARBPL": "L3",
            "EQUNR": "M303",
            "BUDAT": "2025-07-01",
            "SCHICHT": "A",
            "VGW02": 480,
            "ISTDZ": 355,
            "GAMNG": 3100,
            "LMNGA": 2920,
            "XMNGA": 180,
            "MATNR": "SKU-C",
            "AUFNR": "10001235",
            "VIBRATION_RMS": 6.8,
            "TEMP_C": 64.2,
            "MOTOR_CURRENT_A": 14.1,
        },
    ]


def _pm_rows() -> list[dict[str, Any]]:
    return [
        {
            "WERKS": "1000",
            "EQUNR": "M303",
            "TPLNR": "NP-L3-M303",
            "ARBPL": "L3",
            "QMDAT": "2025-07-01",
            "SCHICHT": "A",
            "AUSVN": "2025-07-01T08:15:00",
            "AUSBS": "2025-07-01T09:40:00",
            "AUSZT": 85,
            "FECOD": "CHG01",
            "URCOD": "SETUP",
            "QMTXT": "Changeover",
            "AUFNR": "40001111",
            "QMNUM": "10000001",
        },
        {
            "WERKS": "1000",
            "EQUNR": "M201",
            "TPLNR": "NP-L2-M201",
            "ARBPL": "L2",
            "QMDAT": "2025-07-01",
            "SCHICHT": "B",
            "AUSVN": "2025-07-01T16:02:00",
            "AUSBS": "2025-07-01T16:47:00",
            "AUSZT": 45,
            "FECOD": "BRK02",
            "URCOD": "MECH",
            "QMTXT": "Breakdown — mechanical",
            "AUFNR": "40001112",
            "QMNUM": "10000002",
        },
    ]


def _qm_rows() -> list[dict[str, Any]]:
    return [
        {
            "WERKS": "1000",
            "PRUEFLOS": "30000001",
            "MATNR": "SKU-A",
            "ARBPL": "L1",
            "EQUNR": "M101",
            "BUDAT": "2025-07-01",
            "SCHICHT": "A",
            "LMNGA": 4080,
            "FEHLERANZAHL": 120,
            "FECOD": "DIM",
            "MESSWERT": 10.02,
            "TOLUNTER": 9.5,
            "TOLOBER": 10.5,
            "MERKNR": "0010",
        },
        {
            "WERKS": "1000",
            "PRUEFLOS": "30000002",
            "MATNR": "SKU-C",
            "ARBPL": "L3",
            "EQUNR": "M303",
            "BUDAT": "2025-07-01",
            "SCHICHT": "A",
            "LMNGA": 2920,
            "FEHLERANZAHL": 180,
            "FECOD": "SCRATCH",
            "MESSWERT": 10.18,
            "TOLUNTER": 9.5,
            "TOLOBER": 10.5,
            "MERKNR": "0010",
        },
    ]


def production_template() -> pd.DataFrame:
    return pd.DataFrame(_pp_rows(), columns=PP_COLUMNS)


def downtime_template() -> pd.DataFrame:
    return pd.DataFrame(_pm_rows(), columns=PM_COLUMNS)


def quality_template() -> pd.DataFrame:
    return pd.DataFrame(_qm_rows(), columns=QM_COLUMNS)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def templates_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", README)
        zf.writestr("sap_pp_production.csv", _csv_bytes(production_template()))
        zf.writestr("sap_pm_downtime.csv", _csv_bytes(downtime_template()))
        zf.writestr("sap_qm_quality.csv", _csv_bytes(quality_template()))
    return buf.getvalue()


# Default mapping from these template headers → canonical (used in smoke tests / demos).
TEMPLATE_MAPPINGS: dict[str, dict[str, str]] = {
    "production": {
        "line_id": "ARBPL",
        "machine_id": "EQUNR",
        "shift_date": "BUDAT",
        "shift": "SCHICHT",
        "planned_time_min": "VGW02",
        "run_time_min": "ISTDZ",
        "total_count": "GAMNG",
        "good_count": "LMNGA",
        "reject_count": "XMNGA",
        "vibration_rms": "VIBRATION_RMS",
        "temp_c": "TEMP_C",
        "motor_current_a": "MOTOR_CURRENT_A",
    },
    "downtime": {
        "line_id": "ARBPL",
        "machine_id": "EQUNR",
        "shift_date": "QMDAT",
        "shift": "SCHICHT",
        "start_time": "AUSVN",
        "end_time": "AUSBS",
        "downtime_minutes": "AUSZT",
        "downtime_code": "FECOD",
        "downtime_category": "QMTXT",
    },
    "quality": {
        "line_id": "ARBPL",
        "machine_id": "EQUNR",
        "shift_date": "BUDAT",
        "shift": "SCHICHT",
        "good_count": "LMNGA",
        "reject_count": "FEHLERANZAHL",
        "defect_code": "FECOD",
        "measurement_mm": "MESSWERT",
        "lsl": "TOLUNTER",
        "usl": "TOLOBER",
    },
}
