"""Optional Optuna tuning for scrap / downtime rate forecast & anomaly detection."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from modules.quality_checks import find_col


def _rate_series(df: pd.DataFrame) -> tuple[Optional[pd.Series], str]:
    scrap = find_col(df, "scrap_rate", "reject_rate")
    if scrap:
        return pd.to_numeric(df[scrap], errors="coerce"), "scrap_rate"
    rejects = find_col(df, "reject_count", "rejects")
    total = find_col(df, "total_count", "total_units", "produced_count")
    if rejects and total:
        r = pd.to_numeric(df[rejects], errors="coerce")
        t = pd.to_numeric(df[total], errors="coerce").replace(0, np.nan)
        return (r / t), "scrap_rate_computed"
    dt = find_col(df, "downtime_minutes", "downtime_min")
    planned = find_col(df, "planned_time_min", "planned_minutes")
    if dt and planned:
        d = pd.to_numeric(df[dt], errors="coerce")
        p = pd.to_numeric(df[planned], errors="coerce").replace(0, np.nan)
        return (d / p), "downtime_rate"
    return None, ""


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    # Drop pure targets later; keep engineering features
    return num


def tune_and_forecast(
    df: pd.DataFrame,
    n_trials: int = 20,
    use_optuna: bool = True,
) -> dict[str, Any]:
    """
    Forecast next scrap/downtime rate with a small RandomForest,
    optionally hyperparameter-tuned via Optuna.
    """
    y, yname = _rate_series(df)
    if y is None or y.notna().sum() < 20:
        return {"ok": False, "reason": "Need ≥20 rows with scrap or downtime rate"}

    feats = _feature_matrix(df)
    # Align and drop target-like columns
    drop_cols = [c for c in feats.columns if c.lower() in {yname.lower(), "scrap_rate", "oee"}]
    X = feats.drop(columns=drop_cols, errors="ignore")
    mask = y.notna() & X.notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]
    if len(X) < 20 or X.shape[1] < 1:
        return {"ok": False, "reason": "Insufficient numeric features"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    best_params = {"n_estimators": 80, "max_depth": 6, "min_samples_leaf": 2}
    study_info: dict[str, Any] = {"optuna_used": False}

    if use_optuna:
        try:
            import optuna
            from optuna.samplers import TPESampler

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial: "optuna.Trial") -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 40, 160),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
                }
                model = RandomForestRegressor(random_state=42, **params)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                return float(mean_absolute_error(y_test, pred))

            study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params
            study_info = {
                "optuna_used": True,
                "best_mae": round(float(study.best_value), 5),
                "n_trials": n_trials,
                "best_params": best_params,
            }
        except Exception as exc:
            study_info = {"optuna_used": False, "error": str(exc)}

    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred_test))
    next_pred = float(model.predict(X.tail(1))[0])
    recent = float(y.tail(5).mean())

    # Anomaly detection on rate
    iso = IsolationForest(contamination=0.08, random_state=42)
    rate_vals = y.values.reshape(-1, 1)
    labels = iso.fit_predict(rate_vals)
    anomaly_idx = np.where(labels == -1)[0]
    anomaly_rate = round(100.0 * len(anomaly_idx) / max(len(y), 1), 2)

    return {
        "ok": True,
        "target": yname,
        "mae": round(mae, 5),
        "next_period_forecast": round(next_pred, 4),
        "recent_avg": round(recent, 4),
        "delta_vs_recent": round(next_pred - recent, 4),
        "anomaly_rate_pct": anomaly_rate,
        "anomaly_count": int(len(anomaly_idx)),
        "feature_importance": dict(
            sorted(
                zip(X.columns.tolist(), [round(float(v), 4) for v in model.feature_importances_]),
                key=lambda kv: kv[1],
                reverse=True,
            )[:8]
        ),
        **study_info,
    }
