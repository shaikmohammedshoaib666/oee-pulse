"""Generate synthetic multi-line plant data for OEE Pulse demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DOWNTIME_CODES = [
    ("CHG01", "Changeover", 25, 90),
    ("BRK02", "Breakdown — mechanical", 15, 120),
    ("BRK03", "Breakdown — electrical", 10, 80),
    ("MAT04", "Material shortage", 20, 60),
    ("QLT05", "Quality hold", 10, 45),
    ("OPR06", "Operator absence", 8, 30),
    ("PM07", "Planned maintenance", 30, 180),
    ("STP08", "Minor stoppage", 2, 15),
]


def generate_sample_plant(
    days: int = 21,
    seed: int = 42,
    out_dir: str | Path = "sample_data",
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = ["L1", "L2", "L3"]
    machines = {
        "L1": ["M101", "M102"],
        "L2": ["M201", "M202"],
        "L3": ["M301", "M302", "M303"],
    }
    shifts = ["A", "B"]
    start = pd.Timestamp("2025-07-01")

    prod_rows = []
    dt_rows = []
    q_rows = []
    event_id = 1000

    for d in range(days):
        day = start + pd.Timedelta(days=d)
        for line in lines:
            for machine in machines[line]:
                for shift in shifts:
                    # Line 3 intentionally worse availability (changeovers)
                    base_down = 40 if line != "L3" else 95
                    if machine.endswith("03"):
                        base_down += 25
                    downtime_budget = max(5, rng.normal(base_down, 18))
                    planned = 480.0
                    ideal_rate = float(rng.choice([8, 10, 12]))
                    # Performance noise
                    run = max(planned - downtime_budget, 60)
                    total = int(run * ideal_rate * rng.uniform(0.82, 1.02))
                    scrap_rate = float(np.clip(rng.normal(0.035 if line != "L2" else 0.055, 0.015), 0.005, 0.15))
                    rejects = int(total * scrap_rate)
                    good = total - rejects

                    vib_mu = 6.4 if machine.endswith("03") else (4.8 if line == "L3" else 2.4)
                    temp_mu = 63.0 if line == "L3" else 47.0
                    prod_rows.append(
                        {
                            "shift_date": day.date().isoformat(),
                            "shift": shift,
                            "line_id": line,
                            "machine_id": machine,
                            "planned_time_min": planned,
                            "ideal_rate": ideal_rate,
                            "total_count": total,
                            "product_sku": rng.choice(["SKU-A", "SKU-B", "SKU-C", "SKU-D"]),
                            "operator_id": f"OP{rng.integers(10, 40)}",
                            "vibration_rms": round(float(np.clip(rng.normal(vib_mu, 0.7), 0.4, 12.0)), 3),
                            "temp_c": round(float(np.clip(rng.normal(temp_mu, 4.5), 30.0, 95.0)), 2),
                            "motor_current_a": round(float(np.clip(rng.normal(12.5, 1.6), 6.0, 22.0)), 2),
                        }
                    )

                    defect = str(rng.choice(["DIM", "SCRATCH", "CONTAM", "OTHER"]))
                    defect_type = {
                        "DIM": "Dimension out of spec",
                        "SCRATCH": "Surface scratch",
                        "CONTAM": "Contamination",
                        "OTHER": "Other",
                    }[defect]
                    meas_mu = 10.18 if line == "L2" else 10.02
                    meas_sd = 0.22 if line == "L2" else 0.11
                    q_rows.append(
                        {
                            "shift_date": day.date().isoformat(),
                            "shift": shift,
                            "line_id": line,
                            "machine_id": machine,
                            "good_count": good,
                            "reject_count": rejects,
                            "scrap_rate": round(scrap_rate, 4),
                            "defect_code": defect,
                            "defect_type": defect_type,
                            "measurement_mm": round(float(rng.normal(meas_mu, meas_sd)), 4),
                            "lsl": 9.5,
                            "usl": 10.5,
                        }
                    )

                    # Split downtime budget into 1–4 events
                    remaining = downtime_budget
                    n_events = int(rng.integers(1, 5))
                    for _ in range(n_events):
                        code, category, lo, hi = DOWNTIME_CODES[int(rng.integers(0, len(DOWNTIME_CODES)))]
                        # Bias L3 toward changeovers
                        if line == "L3" and rng.random() < 0.45:
                            code, category, lo, hi = DOWNTIME_CODES[0]
                        mins = float(np.clip(rng.uniform(lo, hi), 1, remaining))
                        remaining -= mins
                        start_hour = 6 if shift == "A" else 14
                        start_ts = day + pd.Timedelta(hours=int(start_hour + rng.integers(0, 7)), minutes=int(rng.integers(0, 50)))
                        end_ts = start_ts + pd.Timedelta(minutes=mins)
                        is_planned = code in {"CHG01", "PM07"}
                        dt_rows.append(
                            {
                                "event_id": event_id,
                                "shift_date": day.date().isoformat(),
                                "shift": shift,
                                "line_id": line,
                                "machine_id": machine,
                                "downtime_code": code,
                                "downtime_category": category,
                                "downtime_minutes": round(mins, 1),
                                "start_time": start_ts.isoformat(),
                                "end_time": end_ts.isoformat(),
                                "planned_time_min": planned,
                                "is_planned": bool(is_planned),
                                "event_class": "Planned" if is_planned else "Unplanned",
                            }
                        )
                        event_id += 1
                        if remaining < 5:
                            break

    production = pd.DataFrame(prod_rows)
    downtime = pd.DataFrame(dt_rows)
    quality = pd.DataFrame(q_rows)

    # Aggregate downtime minutes onto production for convenience
    agg = (
        downtime.groupby(["shift_date", "shift", "line_id", "machine_id"], as_index=False)["downtime_minutes"]
        .sum()
        .rename(columns={"downtime_minutes": "downtime_minutes"})
    )
    production = production.merge(agg, on=["shift_date", "shift", "line_id", "machine_id"], how="left")
    production["downtime_minutes"] = production["downtime_minutes"].fillna(0)

    production.to_csv(out / "production_logs.csv", index=False)
    downtime.to_csv(out / "downtime_events.csv", index=False)
    quality.to_csv(out / "quality_rejects.csv", index=False)

    return {"production": production, "downtime": downtime, "quality": quality}


if __name__ == "__main__":
    data = generate_sample_plant()
    for k, v in data.items():
        print(f"{k}: {v.shape}")
