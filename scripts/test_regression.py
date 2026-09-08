"""Regression guards: previous OEE pipeline still works, Drive ingest is stronger, Cloud pins stay safe."""

from __future__ import annotations

import io
import re
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.data_integration import (
    is_zip_upload,
    load_tabular_file,
    load_upload_for_kind,
    load_zip_tables,
    looks_like_zip_path,
    plant_default_join,
    try_duckdb_join,
)
from modules.oee_engine import oee_summary
from modules.quality_checks import clean_plant_frame
from modules.url_ingest import (
    build_preset_sql,
    detect_source_kind,
    extract_gdrive_file_id,
    friendly_source_label,
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
    assert "Plant ZIP" in app
    assert 'FILE_TYPES = ["csv", "xlsx", "tsv", "json", "parquet", "zip"]' in app
    assert app.count("type=FILE_TYPES") >= 3
    assert "up_prod" in app and "up_dt" in app and "up_q" in app
    assert "load_upload_for_kind" in app
    assert "_load_section_upload" in app


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

    # Browse-box path: CSV still loads through the ZIP-aware helper.
    prod2, meta2 = load_upload_for_kind(sample_dir / "production_logs.csv", "production")
    assert meta2["kind"] == "file"
    assert len(prod2) == len(prod)
    assert "machine_id" in prod2.columns


def _named_upload(name: str, data: bytes):
    buf = io.BytesIO(data)
    buf.name = name
    buf.size = len(data)
    return buf


def _zip_bytes(members: list[tuple[str, Path]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for arcname, path in members:
            zf.write(path, arcname)
    return buf.getvalue()


def _assert_section_zip_upload(sample_dir: Path) -> None:
    """Each browse box accepts ZIP alongside CSV; a plant ZIP in one box does not clobber the others."""
    prod_csv = (sample_dir / "production_logs.csv").read_bytes()
    csv_upload = _named_upload("production_logs.csv", prod_csv)
    assert is_zip_upload(csv_upload) is False
    csv_df, csv_meta = load_upload_for_kind(csv_upload, "production")
    assert csv_meta["kind"] == "file"
    assert len(csv_df) > 50
    # Same buffer can be read again after the ZIP sniff / rewind.
    csv_upload.seek(0)
    again, _ = load_upload_for_kind(csv_upload, "production")
    assert len(again) == len(csv_df)

    tsv_upload = _named_upload(
        "production_logs.tsv",
        csv_df.head(12).to_csv(index=False, sep="\t").encode("utf-8"),
    )
    tsv_df, tsv_meta = load_upload_for_kind(tsv_upload, "production")
    assert tsv_meta["kind"] == "file" and len(tsv_df) == 12

    json_upload = _named_upload(
        "production_logs.json",
        csv_df.head(8).to_json(orient="records", date_format="iso").encode("utf-8"),
    )
    json_df, json_meta = load_upload_for_kind(json_upload, "production")
    assert json_meta["kind"] == "file" and len(json_df) == 8

    plant_members = [
        ("production_logs.csv", sample_dir / "production_logs.csv"),
        ("downtime_events.csv", sample_dir / "downtime_events.csv"),
        ("quality_rejects.csv", sample_dir / "quality_rejects.csv"),
    ]
    plant_zip = _zip_bytes(plant_members)
    assert is_zip_upload(_named_upload("plant.zip", plant_zip)) is True
    # Drive-style ZIP with no .zip suffix still sniffs as ZIP.
    assert is_zip_upload(_named_upload("gdrive_file.bin", plant_zip)) is True

    prod_zip, prod_meta = load_upload_for_kind(_named_upload("plant.zip", plant_zip), "production")
    assert prod_meta["kind"] == "zip"
    assert prod_meta["picked"] == "production"
    assert set(prod_meta["zip_tables"]) == {"production", "downtime", "quality"}
    assert len(prod_zip) == len(csv_df)
    assert "planned_time_min" in prod_zip.columns or "total_count" in prod_zip.columns
    assert "downtime_code" not in prod_zip.columns

    dt_zip, dt_meta = load_upload_for_kind(_named_upload("plant.zip", plant_zip), "downtime")
    assert dt_meta["picked"] == "downtime"
    assert "event_id" in dt_zip.columns or "downtime_minutes" in dt_zip.columns
    assert len(dt_zip) > 50

    q_zip, q_meta = load_upload_for_kind(_named_upload("plant.zip", plant_zip), "quality")
    assert q_meta["picked"] == "quality"
    assert "reject_count" in q_zip.columns or "scrap_rate" in q_zip.columns
    assert len(q_zip) > 50

    # Single-file ZIP in the production box.
    one = _zip_bytes([("production_logs.csv", sample_dir / "production_logs.csv")])
    one_df, one_meta = load_upload_for_kind(_named_upload("prod_only.zip", one), "production")
    assert one_meta["picked"] == "production" and len(one_df) == len(csv_df)

    # Generic single-file ZIP still accepted as the section extract.
    generic = _zip_bytes([("extract.csv", sample_dir / "production_logs.csv")])
    generic_df, generic_meta = load_upload_for_kind(_named_upload("extract.zip", generic), "production")
    assert len(generic_df) == len(csv_df)
    assert generic_meta["kind"] == "zip"

    # Downtime-only ZIP dropped in the production box: only one table, so it is used.
    dt_only = _zip_bytes([("downtime_events.csv", sample_dir / "downtime_events.csv")])
    fallback_df, fallback_meta = load_upload_for_kind(_named_upload("dt.zip", dt_only), "production")
    assert fallback_meta["picked"] == "downtime"
    assert len(fallback_df) == len(dt_zip)

    # Two-file ZIP missing production must fail the production box (do not silently pick the wrong table).
    dt_q = _zip_bytes(
        [
            ("downtime_events.csv", sample_dir / "downtime_events.csv"),
            ("quality_rejects.csv", sample_dir / "quality_rejects.csv"),
        ]
    )
    try:
        load_upload_for_kind(_named_upload("dt_quality.zip", dt_q), "production")
        raise AssertionError("mismatch ZIP should not load as production")
    except ValueError as exc:
        msg = str(exc).lower()
        assert "not production" in msg or "no production" in msg
        assert "plant zip" in msg

    try:
        load_upload_for_kind(_named_upload("plant.zip", plant_zip), "finance")
        raise AssertionError("unknown table kind should fail")
    except ValueError as exc:
        assert "unknown table kind" in str(exc).lower()

    # .xlsx is PK-zipped Office XML — browse boxes must load it as Excel, not a plant ZIP.
    xlsx_buf = io.BytesIO()
    csv_df.head(15).to_excel(xlsx_buf, index=False, engine="openpyxl")
    xlsx_bytes = xlsx_buf.getvalue()
    assert xlsx_bytes[:2] == b"PK"
    xlsx_upload = _named_upload("production_logs.xlsx", xlsx_bytes)
    assert is_zip_upload(xlsx_upload) is False
    xlsx_df, xlsx_meta = load_upload_for_kind(xlsx_upload, "production")
    assert xlsx_meta["kind"] == "file" and len(xlsx_df) == 15

    xlsx_in_zip = io.BytesIO()
    with zipfile.ZipFile(xlsx_in_zip, "w") as zf:
        zf.writestr("production_logs.xlsx", xlsx_bytes)
    xz_df, xz_meta = load_upload_for_kind(_named_upload("prod.zip", xlsx_in_zip.getvalue()), "production")
    assert xz_meta["kind"] == "zip" and len(xz_df) == 15

    with tempfile.TemporaryDirectory(prefix="oee-xlsx-") as tmp:
        xlsx_path = Path(tmp) / "production_logs.xlsx"
        xlsx_path.write_bytes(xlsx_bytes)
        assert looks_like_zip_path(xlsx_path) is False
        path_df, _ = load_upload_for_kind(xlsx_path, "production")
        assert len(path_df) == 15
        unnamed_xlsx = _named_upload("gdrive_file.bin", xlsx_bytes)
        assert is_zip_upload(unnamed_xlsx) is False
        xlsx_loaded, xlsx_url_meta = load_from_url(
            str(xlsx_path), cache_dir=Path(tmp), table_kind="production"
        )
        assert len(xlsx_loaded) == 15
        assert xlsx_url_meta.get("engine") == "pandas-excel"

    # Parquet via DuckDB because Cloud has no pyarrow / fsspec.
    import duckdb

    with tempfile.TemporaryDirectory(prefix="oee-parquet-") as tmp:
        pq_path = Path(tmp) / "production_logs.parquet"
        con = duckdb.connect()
        con.register("prod", csv_df.head(12))
        con.execute("COPY prod TO ? (FORMAT PARQUET)", [str(pq_path)])
        con.close()
        assert is_zip_upload(pq_path) is False
        pq_bytes = pq_path.read_bytes()
        pq_df, pq_meta = load_upload_for_kind(_named_upload("production.parquet", pq_bytes), "production")
        assert pq_meta["kind"] == "file" and len(pq_df) == 12
        import pandas as pandas_mod

        orig = pandas_mod.read_parquet

        def _no_engine(*_a, **_k):
            raise ImportError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.")

        pandas_mod.read_parquet = _no_engine
        try:
            forced, _ = load_upload_for_kind(_named_upload("production.parquet", pq_bytes), "production")
            assert len(forced) == 12
            from_path, _ = load_upload_for_kind(pq_path, "production")
            assert len(from_path) == 12
        finally:
            pandas_mod.read_parquet = orig

    assert friendly_source_label({"kind": "zip", "zip_name": "plant.zip"}) == "plant.zip"
    assert friendly_source_label({"kind": "file", "zip_name": "production_logs.csv"}) == "production_logs.csv"


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


def _assert_zip_upload(sample_dir: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(sample_dir / "production_logs.csv", "plant/production_logs.csv")
        zf.write(sample_dir / "downtime_events.csv", "nested/downtime_events.csv")
        zf.write(sample_dir / "quality_rejects.csv", "quality_rejects.csv")
        zf.writestr("__MACOSX/._junk.csv", "skip,me\n1,2\n")
        zf.writestr("README.txt", "ignore me")
    buf.seek(0)
    tables, log = load_zip_tables(buf)
    assert set(tables) == {"production", "downtime", "quality"}
    assert len(tables["production"]) > 50
    kinds = {m["kind"] for m in log if m["kind"] != "unclassified"}
    assert "production" in kinds and "downtime" in kinds and "quality" in kinds

    from modules.sap_templates import templates_zip_bytes

    sap_tables, sap_log = load_zip_tables(io.BytesIO(templates_zip_bytes()))
    assert set(sap_tables) >= {"production", "downtime", "quality"}
    assert any("sap_pp" in (m.get("member") or "") for m in sap_log)

    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.csv", "a,b\n1,2\n")
    evil.seek(0)
    try:
        load_zip_tables(evil)
        raise AssertionError("zip-slip path should be rejected")
    except ValueError:
        pass

    named = io.BytesIO()
    with zipfile.ZipFile(named, "w") as zf:
        zf.write(sample_dir / "production_logs.csv", "plant.zip")  # not tabular suffix
    named.seek(0)
    # file named plant.zip inside zip is skipped; expect empty/error
    try:
        load_zip_tables(named)
        raise AssertionError("non-tabular zip should fail")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory(prefix="oee-zip-") as tmp:
        zpath = Path(tmp) / "plant.zip"
        buf.seek(0)
        zpath.write_bytes(buf.getvalue())
        sliced, meta = load_from_url(
            str(zpath),
            cache_dir=Path(tmp),
            table_kind="production",
            sql_query=build_preset_sql("first_n_rows", {"n": 20}),
        )
        assert len(sliced) == 20
        assert "production" in (meta.get("zip_tables") or [])
        extra = meta.get("extra_tables") or {}
        assert "downtime" in extra and "quality" in extra

    # Rewind: same buffer can be opened twice.
    buf.seek(0)
    again, _ = load_zip_tables(buf)
    assert set(again) == {"production", "downtime", "quality"}

    # Magic-byte ZIP without a .zip suffix (Drive often caches as .csv).
    buf.seek(0)
    with tempfile.TemporaryDirectory(prefix="oee-zip-magic-") as tmp:
        magic_path = Path(tmp) / "gdrive_file.bin"
        magic_path.write_bytes(buf.getvalue())
        sniffed, sniff_meta = load_from_url(
            str(magic_path),
            cache_dir=Path(tmp),
            table_kind="downtime",
            row_limit=8,
        )
        assert sniff_meta.get("engine", "").startswith("zip")
        assert len(sniffed) == 8
        assert "production" in (sniff_meta.get("extra_tables") or {})

    # One corrupt member must not kill the rest of the archive.
    mixed = io.BytesIO()
    with zipfile.ZipFile(mixed, "w") as zf:
        zf.write(sample_dir / "production_logs.csv", "production_logs.csv")
        zf.writestr("downtime_events.csv", b"\xff\xfe not,a,csv")
        zf.write(sample_dir / "quality_rejects.csv", "quality_rejects.csv")
    mixed.seek(0)
    mixed_tables, mixed_log = load_zip_tables(mixed)
    assert "production" in mixed_tables and "quality" in mixed_tables
    assert any(m.get("kind") == "error" for m in mixed_log)

    # ZIP tables still run the original OEE path.
    frame = tables["production"].merge(
        tables["quality"], on=["shift_date", "shift", "line_id", "machine_id"], how="left"
    )
    cleaned, _ = clean_plant_frame(frame)
    plant = oee_summary(cleaned)["plant"]
    assert 0 <= float(plant["oee"]) <= 1.5


def run_regression(sample_dir: Path) -> None:
    _assert_cloud_safe_contract()
    _assert_file_upload_still_works(sample_dir)
    _assert_section_zip_upload(sample_dir)
    _assert_zip_upload(sample_dir)
    _assert_drive_urls_and_sql_guards()
    _assert_sliced_ingest_feeds_oee(sample_dir)
    _assert_app_source_guards()


if __name__ == "__main__":
    from modules.sample_data import generate_sample_plant

    out = ROOT / "sample_data"
    generate_sample_plant(out_dir=out)
    run_regression(out)
    print("REGRESSION OK")
