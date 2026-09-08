"""Regression guards: previous OEE pipeline still works, Drive ingest is stronger, Cloud pins stay safe."""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.data_integration import load_tabular_file, plant_default_join, try_duckdb_join
from modules.oee_engine import oee_summary
from modules.quality_checks import clean_plant_frame
from modules.url_ingest import (
    build_preset_sql,
    detect_source_kind,
    extract_gdrive_file_id,
    load_from_url,
    validate_ingest_sql,
)


def _assert_cloud_safe_contract() -> None:
    """The Streamlit Cloud crash was unpinned latest packages + Python 3.13. Lock the contract."""
    req = (ROOT / "requirements.txt").read_text()
    assert "pandas>=2.2.0,<3.0.0" in req, "pandas must stay 2.x for Community Cloud"
    assert "streamlit>=1.32.0,<1.48.0" in req, "Streamlit must stay below 1.48 on Cloud"
    assert "plotly>=5.18.0,<6.0.0" in req
    assert "openai>=1.40.0,<2.0.0" in req
    assert "pyarrow" not in req, "pyarrow bloated Cloud installs; DuckDB covers parquet"
    runtime = (ROOT / "runtime.txt").read_text().strip()
    assert runtime == "python-3.12"
    cfg = (ROOT / ".streamlit" / "config.toml").read_text()
    assert "[theme]" in cfg
    assert "headless" not in cfg
    app = (ROOT / "app.py").read_text()
    assert "URL / Drive (DuckDB)" in app
    assert "_URL_INGEST_ERROR" in app
    assert "File upload" in app


def _assert_file_upload_still_works(sample_dir: Path) -> None:
    """Previous Upload path: local CSV → map-ready frame → join → OEE."""
    prod = load_tabular_file(sample_dir / "production_logs.csv")
    dt = load_tabular_file(sample_dir / "downtime_events.csv")
    qual = load_tabular_file(sample_dir / "quality_rejects.csv")
    assert len(prod) > 50 and len(dt) > 50 and len(qual) > 50
    for col in ("machine_id", "line_id", "shift_date", "shift"):
        assert col in prod.columns and col in dt.columns and col in qual.columns

    joined, logs = plant_default_join(prod, dt, qual)
    assert logs and len(joined) > 0
    duck = try_duckdb_join(prod, dt, qual)
    assert duck is not None and len(duck) > 0

    frame = prod.merge(qual, on=["shift_date", "shift", "line_id", "machine_id"], how="left")
    cleaned, _ = clean_plant_frame(frame)
    plant = oee_summary(cleaned)["plant"]
    assert 0 <= float(plant["oee"]) <= 1.5
    assert float(plant["availability"]) > 0


def _assert_drive_urls_and_sql_guards() -> None:
    samples = {
        "https://drive.google.com/file/d/1AbCDefGhiJKLmnopQRstuVWxyz012345/view?usp=sharing": "1AbCDefGhiJKLmnopQRstuVWxyz012345",
        "https://drive.google.com/open?id=abcXYZ99": "abcXYZ99",
        "https://drive.google.com/uc?export=download&id=fileId001": "fileId001",
        "https://docs.google.com/spreadsheets/d/sheetID99/edit#gid=0": "sheetID99",
        "https://drive.usercontent.google.com/download?id=ucid88&export=download": "ucid88",
    }
    for url, fid in samples.items():
        assert detect_source_kind(url) == "google_drive"
        assert extract_gdrive_file_id(url) == fid
    assert detect_source_kind("https://example.com/plant.csv") == "https"
    assert detect_source_kind("") == "empty"

    for bad in (
        "DROP TABLE production",
        "DELETE FROM production",
        "INSERT INTO production VALUES (1)",
        "COPY production TO 'x.csv'",
        "SELECT * FROM read_csv_auto('x.csv'); SELECT 1",
        "SELECT * FROM read_csv_auto('x.csv')",
    ):
        try:
            validate_ingest_sql(bad)
        except ValueError:
            continue
        raise AssertionError(f"unsafe SQL was accepted: {bad}")


def _assert_sliced_ingest_feeds_oee(sample_dir: Path) -> None:
    """New Drive/SQL path must still produce valid OEE — better than loading 2 GB into pandas."""
    prod_path = sample_dir / "production_logs.csv"
    dt_path = sample_dir / "downtime_events.csv"
    q_path = sample_dir / "quality_rejects.csv"
    with tempfile.TemporaryDirectory(prefix="oee-reg-") as tmp:
        cache = Path(tmp)
        prod, _ = load_from_url(
            str(prod_path),
            cache_dir=cache,
            sql_query=build_preset_sql(
                "date_range",
                {"start_date": "2025-07-01", "end_date": "2025-07-15", "ts_col": "shift_date"},
            ),
        )
        dt, _ = load_from_url(
            str(dt_path),
            cache_dir=cache,
            sql_query=build_preset_sql(
                "between_ids",
                {"id_col": "event_id", "start_id": "1000", "end_id": "1500"},
            ),
        )
        qual, _ = load_from_url(
            str(q_path),
            cache_dir=cache,
            sql_query=build_preset_sql("filter_line_id", {"line_id": "L1", "n": 0}),
        )
    assert len(prod) > 0 and len(dt) > 0 and len(qual) > 0
    assert set(qual["line_id"].astype(str).unique()) == {"L1"}

    frame = prod.merge(qual, on=["shift_date", "shift", "line_id", "machine_id"], how="left")
    cleaned, _ = clean_plant_frame(frame)
    plant = oee_summary(cleaned)["plant"]
    assert 0 <= float(plant["oee"]) <= 1.5
    assert "availability" in plant and "performance" in plant and "quality" in plant


def _assert_app_source_guards() -> None:
    """Widget defaults must not use index= + key= together on the URL ingest radio/selectbox."""
    app = (ROOT / "app.py").read_text()
    radio = re.search(
        r"ingest_mode = st\.radio\((.*?)key=\"upload_url_ingest_mode\"",
        app,
        re.S,
    )
    assert radio, "URL ingest mode radio missing"
    assert "index=" not in radio.group(1)
    table = re.search(
        r"table_kind = st\.selectbox\((.*?)key=\"url_ingest_table_pick\"",
        app,
        re.S,
    )
    assert table, "URL ingest table selectbox missing"
    assert "index=" not in table.group(1)


def run_regression(sample_dir: Path) -> None:
    _assert_cloud_safe_contract()
    _assert_file_upload_still_works(sample_dir)
    _assert_drive_urls_and_sql_guards()
    _assert_sliced_ingest_feeds_oee(sample_dir)
    _assert_app_source_guards()


if __name__ == "__main__":
    from modules.sample_data import generate_sample_plant

    out = ROOT / "sample_data"
    generate_sample_plant(out_dir=out)
    run_regression(out)
    print("REGRESSION OK")
