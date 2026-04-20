#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regime-Aware Online Model Prediction Set Backtest — Final Corrected Version
===========================================================================

Main idea
---------
This script implements a pragmatic, paper-ready backtest for:
1. Next-day realized variance / volatility forecasting
2. Candidate model pool:
   - HAR-RV
   - GARCH-t
   - FIGARCH
   - GRU
   - XGBoost (with sklearn fallback)
   - EGARCH optional, dropped by default
3. Route-B risk-sensitive best-model definition:
   L = QLIKE + lambda_under * UnderPenalty
4. Rolling-best and static-best baselines
5. Naive VIX-switch baseline:
   - VIX > threshold -> GARCH-t
   - else -> GRU
6. Regime-aware online model scoring with local-shrunk tau calibration
7. Two-sided state-gated blend policy:
   - low-state branch blends rolling_best and calm specialist
   - high-state branch blends combo and stress specialist
   - omega_t blends low/high branches
   - HAR acts as a conditional floor in stressed / high-dispersion states
8. Regime-conditional aggregation:
   - calm specialist pool: GRU / HAR / XGB
   - stress specialist pool: GARCH-t / FIGARCH / HAR
9. Routing diagnostics:
   - compact 4-metric regime table
   - detailed specialist day-usage table
10. Cross-asset Diebold–Mariano tests with Newey–West HAC variance

Important note
--------------
QLIKE is most naturally defined on variance-like targets. Therefore this
implementation uses next-day realized variance as the primary target:
    y_{t+1} = (gk_proxy_{t+1})^2
and reports RMSE / MAE on the corresponding volatility scale sqrt(y).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import os
# Optional dependencies
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False


# =========================================================
# Config
# =========================================================
from pathlib import Path
from dataclasses import dataclass

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

@dataclass
class Config:
    data_dir: str = os.environ.get("REGIME_ROUTING_DATA_DIR", str(DEFAULT_DATA_DIR))
    results_dir: str = os.environ.get("REGIME_ROUTING_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))

    panel_file: str = "multiasset_daily_10y_panel_model.csv"
    macro_file: str = "master_daily_features_macro_dailyonly_raw_only.csv"

    symbol: str = "SPY"

    # target / feature choices
    target_proxy: str = "gk_proxy"
    use_macro: bool = True
    eps: float = 1e-10

    # walk-forward
    train_window: int = 504
    compare_window: int = 252
    set_calibration_window: int = 252
    min_train_history: int = 504
    refit_every: int = 21
    min_eval_history_for_set: int = 252
    static_selection_window: int = 252

    # risk-sensitive best-model loss
    lambda_under: float = 1.0

    # regime-aware weighting
    time_half_life: float = 63.0
    regime_kappa: float = 1.5
    score_history_window: int = 252
    alpha: float = 0.10

    # sequence / ML model
    seq_len: int = 20
    gru_hidden_dim: int = 16
    gru_num_layers: int = 1
    gru_dropout: float = 0.0
    gru_epochs: int = 12
    gru_batch_size: int = 64
    gru_lr: float = 1e-3
    use_cuda_if_available: bool = True

    # xgboost
    xgb_n_estimators: int = 250
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0

    # pool / policy
    egarch_mode: str = "drop"
    calm_top_m_cap: int = 1
    stress_top_m_cap: int = 2
    stress_prob_threshold: float = 0.55

    # naive VIX-switch baseline
    vix_switch_threshold: float = 20.0

    # stress specialist aggregation
    upper_quantile_q: float = 0.75
    winsor_lower_q: float = 0.10
    winsor_upper_q: float = 0.90

    # local / shrunk tau
    local_tau_regime_kappa: float = 1.25
    local_tau_shrinkage: float = 8.0
    min_local_tau_eff_n: float = 10.0
    max_local_tau_weight: float = 0.80

    # stress gate
    stress_gate_center: float = 0.20
    stress_gate_scale: float = 1.00

    # two-sided blend policy
    low_blend_rho: float = 0.25
    high_blend_kappa: float = 0.35
    blend_stress_midpoint: float = 0.55
    blend_stress_scale: float = 0.12

    # conditional HAR floor
    har_floor_stress_threshold: float = 0.65
    har_floor_dispersion_threshold: float = 0.20

    # misc
    verbose: bool = True
    seed: int = 42
    force_cpu: bool = False


# =========================================================
# Utility
# =========================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def qlike_loss(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    y = np.maximum(y, eps)
    yhat = np.maximum(yhat, eps)
    ratio = y / yhat
    return ratio - np.log(ratio) - 1.0


def underprediction_penalty(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    num = np.maximum(y - yhat, 0.0)
    den = np.maximum(y, eps)
    return (num / den) ** 2


def composite_loss(y: np.ndarray, yhat: np.ndarray, lambda_under: float, eps: float) -> np.ndarray:
    return qlike_loss(y, yhat, eps=eps) + lambda_under * underprediction_penalty(y, yhat, eps=eps)


def safe_mean(x: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) > 0 else np.nan


def time_decay_weights(lags: np.ndarray, half_life: float) -> np.ndarray:
    half_life = max(float(half_life), 1e-6)
    return np.exp(-np.log(2.0) * lags / half_life)


def jaccard_distance(set_a: List[str], set_b: List[str]) -> float:
    a, b = set(set_a), set(set_b)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    return 1.0 - len(a & b) / max(len(a | b), 1)


def winsorize_array(x: np.ndarray, lower_q: float, upper_q: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return x
    lower_q = float(np.clip(lower_q, 0.0, 1.0))
    upper_q = float(np.clip(upper_q, 0.0, 1.0))
    if upper_q < lower_q:
        upper_q = lower_q
    lo = float(np.quantile(x, lower_q))
    hi = float(np.quantile(x, upper_q))
    return np.clip(x, lo, hi)


def weighted_quantile(values: np.ndarray, quantile: float, sample_weight: Optional[np.ndarray] = None) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if sample_weight is None:
        sample_weight = np.ones(len(values), dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    mask &= np.isfinite(sample_weight) & (sample_weight > 0)
    values = values[mask]
    sample_weight = sample_weight[mask]
    if len(values) == 0:
        return np.nan
    if len(values) == 1:
        return float(values[0])
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    cum_w = np.cumsum(sample_weight)
    total_w = cum_w[-1]
    if total_w <= 0:
        return np.nan
    q = float(np.clip(quantile, 0.0, 1.0))
    cutoff = q * total_w
    idx = int(np.searchsorted(cum_w, cutoff, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if len(weights) == 0:
        return 0.0
    sw = np.sum(weights)
    sw2 = np.sum(weights ** 2)
    if sw2 <= 0:
        return 0.0
    return float((sw ** 2) / sw2)


def robust_relative_dispersion(vals: np.ndarray, eps: float = 1e-12) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return 0.0
    q25, q75 = np.quantile(vals, [0.25, 0.75])
    med = float(np.median(vals))
    return float((q75 - q25) / max(abs(med), eps))


def get_model_pools(cfg: Config) -> Tuple[List[str], List[str]]:
    all_model_names = ["har", "garch_t", "egarch", "figarch", "gru", "xgb"]
    if cfg.egarch_mode not in {"drop", "active"}:
        raise ValueError("egarch_mode must be one of {'drop', 'active'}.")
    if cfg.egarch_mode == "drop":
        stored_model_names = [m for m in all_model_names if m != "egarch"]
        active_model_names = list(stored_model_names)
    else:
        stored_model_names = list(all_model_names)
        active_model_names = list(all_model_names)
    return stored_model_names, active_model_names


def regime_name_map(x: int | float) -> str:
    if x == 0:
        return "low"
    if x == 1:
        return "mid"
    if x == 2:
        return "high"
    return "unknown"


# =========================================================
# Diebold–Mariano test
# =========================================================
def newey_west_long_run_variance(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= 1:
        return np.nan
    lag = max(int(lag), 0)
    mu = float(np.mean(x))
    xc = x - mu
    gamma0 = float(np.dot(xc, xc) / n)
    lrvar = gamma0
    for h in range(1, min(lag, n - 1) + 1):
        gamma_h = float(np.dot(xc[h:], xc[:-h]) / n)
        weight = 1.0 - h / (lag + 1.0)
        lrvar += 2.0 * weight * gamma_h
    return max(lrvar, 1e-20)


def diebold_mariano_test(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss_name: str = "qlike",
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    d_t = L_a - L_b
    Negative mean / negative stat => method_a has lower loss.
    """
    y = np.asarray(y, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    mask = np.isfinite(y) & np.isfinite(pred_a) & np.isfinite(pred_b)
    y = y[mask]
    pred_a = pred_a[mask]
    pred_b = pred_b[mask]
    n = len(y)

    if n < 30:
        return {
            "n_obs": n,
            "mean_loss_a": np.nan,
            "mean_loss_b": np.nan,
            "mean_diff_a_minus_b": np.nan,
            "nw_lag": np.nan,
            "dm_stat": np.nan,
            "p_value": np.nan,
            "reject_5pct": np.nan,
            "favors_a": np.nan,
        }

    if loss_name == "qlike":
        loss_a = qlike_loss(y, pred_a, eps=eps)
        loss_b = qlike_loss(y, pred_b, eps=eps)
    elif loss_name == "se":
        loss_a = (y - pred_a) ** 2
        loss_b = (y - pred_b) ** 2
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    d = loss_a - loss_b
    dbar = float(np.mean(d))
    nw_lag = max(int(np.ceil(n ** (1.0 / 3.0))), 1)
    lrvar = newey_west_long_run_variance(d, lag=nw_lag)

    if not np.isfinite(lrvar) or lrvar <= 0:
        stat = np.nan
        pval = np.nan
    else:
        stat = dbar / math.sqrt(lrvar / n)
        pval = 2.0 * (1.0 - normal_cdf(abs(stat)))

    favors_a = float((np.isfinite(stat) and stat < 0.0))
    reject = float((np.isfinite(pval) and pval < 0.05))

    return {
        "n_obs": n,
        "mean_loss_a": float(np.mean(loss_a)),
        "mean_loss_b": float(np.mean(loss_b)),
        "mean_diff_a_minus_b": dbar,
        "nw_lag": float(nw_lag),
        "dm_stat": float(stat) if np.isfinite(stat) else np.nan,
        "p_value": float(pval) if np.isfinite(pval) else np.nan,
        "reject_5pct": reject,
        "favors_a": favors_a,
    }


def format_dm_summary_sentence(dm_df: pd.DataFrame, method_a: str, method_b: str) -> str:
    if dm_df.empty:
        return ""
    use = dm_df[(dm_df["method_a"] == method_a) & (dm_df["method_b"] == method_b)].copy()
    if use.empty:
        return ""

    use = use.sort_values("asset")
    favorable = use[(use["reject_5pct"] == 1.0) & (use["favors_a"] == 1.0)]
    favorable_assets = favorable["asset"].astype(str).tolist()

    other = use[~((use["reject_5pct"] == 1.0) & (use["favors_a"] == 1.0))][["asset", "p_value", "dm_stat"]].copy()

    n_good = len(favorable_assets)
    n_total = len(use)

    if len(other) > 0:
        tail = "; ".join(
            [
                f"{r['asset']} p={r['p_value']:.3f}, stat={r['dm_stat']:.3f}"
                if pd.notna(r["p_value"]) and pd.notna(r["dm_stat"])
                else f"{r['asset']} p=NA"
                for _, r in other.iterrows()
            ]
        )
        if n_good > 0:
            return (
                f"Diebold–Mariano tests favor {method_a} over {method_b} at the 5% level for "
                f"{n_good}/{n_total} assets ({', '.join(favorable_assets)}); {tail}."
            )
        return (
            f"Diebold–Mariano tests do not favor {method_a} over {method_b} at the 5% level; "
            f"{tail}."
        )

    return (
        f"Diebold–Mariano tests favor {method_a} over {method_b} at the 5% level for "
        f"all {n_total} assets ({', '.join(favorable_assets)})."
    )


# =========================================================
# Data prep
# =========================================================
def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(cfg.data_dir)

    panel_path = data_dir / cfg.panel_file
    macro_path = data_dir / cfg.macro_file

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")

    if not macro_path.exists():
        raise FileNotFoundError(f"Macro file not found: {macro_path}")

    panel = pd.read_csv(panel_path, parse_dates=["date"])
    macro = pd.read_csv(macro_path, parse_dates=["date"])
    return panel, macro


def prepare_symbol_dataset(panel: pd.DataFrame, macro: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = panel.copy()
    df = df[df["symbol"] == cfg.symbol].sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Symbol {cfg.symbol} not found in panel file.")
    if cfg.target_proxy not in df.columns:
        raise KeyError(f"target_proxy {cfg.target_proxy} not found in panel data.")

    df["rv_var"] = np.maximum(df[cfg.target_proxy].astype(float).values, cfg.eps) ** 2
    df["rv_vol"] = np.sqrt(df["rv_var"])
    df["y_next_var"] = df["rv_var"].shift(-1)
    df["y_next_vol"] = np.sqrt(df["y_next_var"])

    for lag in [1, 2, 3, 5, 10, 22]:
        df[f"rv_var_lag{lag}"] = df["rv_var"].shift(lag)
        df[f"ret_lag{lag}"] = df["ret"].shift(lag)
        df[f"absret_lag{lag}"] = df["ret"].abs().shift(lag)

    df["har_daily"] = df["rv_var"].shift(0)
    df["har_weekly"] = df["rv_var"].rolling(5).mean().shift(0)
    df["har_monthly"] = df["rv_var"].rolling(22).mean().shift(0)

    df["rv_mean_5"] = df["rv_var"].rolling(5).mean().shift(0)
    df["rv_mean_10"] = df["rv_var"].rolling(10).mean().shift(0)
    df["rv_mean_22"] = df["rv_var"].rolling(22).mean().shift(0)

    df["rv_std_5"] = df["rv_var"].rolling(5).std().shift(0)
    df["rv_std_22"] = df["rv_var"].rolling(22).std().shift(0)

    df["log_rv_var"] = np.log(np.maximum(df["rv_var"], cfg.eps))
    df["log_rv_var_lag1"] = df["log_rv_var"].shift(1)
    df["vol_of_vol_20"] = df["rv_vol"].rolling(20).std().shift(0)
    df["ret_sq"] = df["ret"] ** 2

    # ---- Missing volatility features: now explicitly created ----
    df["rolling_vol_5"] = df["ret"].rolling(5).std().shift(0)
    df["rolling_vol_10"] = df["ret"].rolling(10).std().shift(0)
    df["rolling_vol_20"] = df["ret"].rolling(20).std().shift(0)
    df["ewma_vol_20"] = df["ret"].ewm(span=20, adjust=False).std().shift(0)

    macro_use = macro.copy().sort_values("date")
    if "vix" not in macro_use.columns or "vxv" not in macro_use.columns:
        raise KeyError("Macro file must contain vix and vxv columns.")
    if "rate_10y" not in macro_use.columns or "rate_2y" not in macro_use.columns:
        raise KeyError("Macro file must contain rate_10y and rate_2y columns.")

    macro_use["log_vix"] = np.log(np.maximum(macro_use["vix"].astype(float), cfg.eps))
    macro_use["log_vix_vxv"] = np.log(
        np.maximum(macro_use["vix"].astype(float), cfg.eps) /
        np.maximum(macro_use["vxv"].astype(float), cfg.eps)
    )
    macro_use["d5_log_vix"] = macro_use["log_vix"].diff(5)
    macro_use["term_spread"] = macro_use["rate_10y"].astype(float) - macro_use["rate_2y"].astype(float)
    macro_use["hy_oas"] = pd.to_numeric(macro_use.get("hy_oas", np.nan), errors="coerce")
    macro_use["vix_raw"] = pd.to_numeric(macro_use["vix"], errors="coerce")

    macro_use = macro_use[
        ["date", "log_vix", "log_vix_vxv", "d5_log_vix", "term_spread", "hy_oas", "vix_raw"]
    ]
    df = df.merge(macro_use, on="date", how="left")

    df["rv20d"] = df["rv_vol"].rolling(20).mean().shift(0)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def make_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    tabular_features = [
        "har_daily", "har_weekly", "har_monthly",
        "rv_var_lag1", "rv_var_lag2", "rv_var_lag5", "rv_var_lag10", "rv_var_lag22",
        "rv_mean_5", "rv_mean_10", "rv_mean_22",
        "rv_std_5", "rv_std_22",
        "ret", "absret_lag1", "absret_lag2", "absret_lag5", "ret_sq",
        "rolling_vol_5", "rolling_vol_10", "rolling_vol_20", "ewma_vol_20",
        "log_vix", "log_vix_vxv", "d5_log_vix", "term_spread", "hy_oas",
        "rv20d", "vol_of_vol_20",
    ]
    tabular_features = [c for c in tabular_features if c in df.columns]

    har_features = [c for c in ["har_daily", "har_weekly", "har_monthly"] if c in df.columns]

    seq_features = [
        "log_rv_var", "ret", "rolling_vol_20", "ewma_vol_20",
        "log_vix", "log_vix_vxv", "d5_log_vix", "term_spread", "hy_oas",
        "rv20d", "vol_of_vol_20",
    ]
    seq_features = [c for c in seq_features if c in df.columns]
    return tabular_features, har_features, seq_features


def build_regime_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    regime_cols = [c for c in ["log_vix", "log_vix_vxv", "rv20d", "vol_of_vol_20", "term_spread", "hy_oas"] if c in df.columns]
    if len(regime_cols) == 0:
        raise ValueError("No regime columns available.")
    return df[regime_cols].values.astype(float), regime_cols


# =========================================================
# Models
# =========================================================
class HARModel:
    def __init__(self) -> None:
        self.model = LinearRegression()
        self.fitted = False
        self.features: List[str] = []

    def fit(self, df_train: pd.DataFrame, features: List[str], target_col: str) -> None:
        self.features = list(features)
        use = df_train[self.features + [target_col]].dropna()
        if len(use) < 30:
            self.fitted = False
            return
        X = use[self.features].values
        y = use[target_col].values
        self.model.fit(X, y)
        self.fitted = True

    def predict_row(self, row: pd.Series, fallback: float) -> float:
        if not self.fitted:
            return float(fallback)
        X = row[self.features].values.astype(float).reshape(1, -1)
        if not np.isfinite(X).all():
            return float(fallback)
        pred = float(self.model.predict(X)[0])
        return max(pred, 1e-12)


class XGBModelWrapper:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model = None
        self.fitted = False
        self.features: List[str] = []

    def fit(self, df_train: pd.DataFrame, features: List[str], target_col: str) -> None:
        self.features = list(features)
        use = df_train[self.features + [target_col]].dropna()
        if len(use) < 80:
            self.fitted = False
            return
        X = use[self.features].values
        y = use[target_col].values
        if HAS_XGB:
            self.model = XGBRegressor(
                n_estimators=self.cfg.xgb_n_estimators,
                max_depth=self.cfg.xgb_max_depth,
                learning_rate=self.cfg.xgb_learning_rate,
                subsample=self.cfg.xgb_subsample,
                colsample_bytree=self.cfg.xgb_colsample_bytree,
                reg_lambda=self.cfg.xgb_reg_lambda,
                objective="reg:squarederror",
                random_state=self.cfg.seed,
                n_jobs=1,
            )
        else:
            self.model = GradientBoostingRegressor(random_state=self.cfg.seed)
        self.model.fit(X, y)
        self.fitted = True

    def predict_row(self, row: pd.Series, fallback: float) -> float:
        if not self.fitted:
            return float(fallback)
        X = row[self.features].values.astype(float).reshape(1, -1)
        if not np.isfinite(X).all():
            return float(fallback)
        pred = float(self.model.predict(X)[0])
        return max(pred, 1e-12)


if HAS_TORCH:
    class GRUNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
            super().__init__()
            self.gru = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)
else:
    GRUNet = object


class GRUModelWrapper:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.net: Optional[GRUNet] = None
        self.seq_features: List[str] = []
        self.fitted = False
        if HAS_TORCH:
            use_cuda = torch.cuda.is_available() and cfg.use_cuda_if_available and not cfg.force_cpu
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = None

    def _make_sequence_dataset(self, df: pd.DataFrame, seq_features: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        vals = df[seq_features + [target_col]].copy()
        for i in range(self.cfg.seq_len - 1, len(vals)):
            block = vals.iloc[i - self.cfg.seq_len + 1:i + 1]
            y = vals.iloc[i][target_col]
            if block[seq_features].isna().any().any() or not np.isfinite(y):
                continue
            X_list.append(block[seq_features].values.astype(float))
            y_list.append(float(y))
        if len(X_list) == 0:
            return np.empty((0, self.cfg.seq_len, len(seq_features))), np.empty((0,))
        return np.asarray(X_list), np.asarray(y_list)

    def fit(self, df_train: pd.DataFrame, seq_features: List[str], target_col: str) -> None:
        self.seq_features = list(seq_features)
        if not HAS_TORCH:
            self.fitted = False
            return

        X, y = self._make_sequence_dataset(df_train, self.seq_features, target_col)
        if len(X) < 80:
            self.fitted = False
            return

        n, s, d = X.shape
        X2 = X.reshape(n * s, d)
        self.scaler_x.fit(X2)
        Xs = self.scaler_x.transform(X2).reshape(n, s, d)

        y_log = np.log(np.maximum(y, self.cfg.eps))
        self.scaler_y.fit(y_log.reshape(-1, 1))
        ys = self.scaler_y.transform(y_log.reshape(-1, 1)).reshape(-1)

        x_tensor = torch.tensor(Xs, dtype=torch.float32)
        y_tensor = torch.tensor(ys, dtype=torch.float32)
        ds = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(ds, batch_size=self.cfg.gru_batch_size, shuffle=True)

        self.net = GRUNet(d, self.cfg.gru_hidden_dim, self.cfg.gru_num_layers, self.cfg.gru_dropout).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.gru_lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for _ in range(self.cfg.gru_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        self.fitted = True

    def predict_row(self, hist_df_up_to_t: pd.DataFrame, fallback: float) -> float:
        if not self.fitted or self.net is None or not HAS_TORCH:
            return float(fallback)
        if len(hist_df_up_to_t) < self.cfg.seq_len:
            return float(fallback)

        block = hist_df_up_to_t[self.seq_features].iloc[-self.cfg.seq_len:].copy()
        if block.isna().any().any():
            return float(fallback)

        X = block.values.astype(float)
        Xs = self.scaler_x.transform(X).reshape(1, self.cfg.seq_len, len(self.seq_features))
        xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)

        self.net.eval()
        with torch.no_grad():
            pred_scaled = float(self.net(xt).cpu().numpy().reshape(-1)[0])

        pred_logvar = self.scaler_y.inverse_transform(np.array([[pred_scaled]])).reshape(-1)[0]
        pred = float(np.exp(pred_logvar))
        return max(pred, 1e-12)


def fit_predict_arch_variant(returns_train: np.ndarray, variant: str, fallback_var: float) -> float:
    if not HAS_ARCH:
        return float(fallback_var)

    r = np.asarray(returns_train, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 120:
        return float(fallback_var)

    r_pct = 100.0 * r
    try:
        if variant == "garch_t":
            am = arch_model(r_pct, mean="Zero", vol="GARCH", p=1, o=0, q=1, dist="t")
        elif variant == "egarch":
            am = arch_model(r_pct, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="t")
        elif variant == "figarch":
            am = arch_model(r_pct, mean="Zero", vol="FIGARCH", p=1, q=1, dist="t")
        else:
            raise ValueError(f"Unknown ARCH variant: {variant}")

        res = am.fit(disp="off", show_warning=False, update_freq=0)
        f = res.forecast(horizon=1, reindex=False)
        var_pct2 = float(f.variance.values[-1, 0])
        var_dec2 = max(var_pct2 / 10000.0, 1e-12)
        return var_dec2
    except Exception:
        return float(fallback_var)


# =========================================================
# Online model set logic
# =========================================================
def compute_model_scores(
    current_pos: int,
    prior_positions: np.ndarray,
    regret_matrix_active: np.ndarray,
    regime_z: np.ndarray,
    cfg: Config,
    mode: str,
) -> np.ndarray:
    n_models = regret_matrix_active.shape[1]
    scores = np.full(n_models, np.nan)
    if len(prior_positions) == 0:
        return scores

    hist_positions = prior_positions[-cfg.score_history_window:]
    hist_regrets = regret_matrix_active[hist_positions, :]
    valid_row = np.isfinite(hist_regrets).all(axis=1)
    hist_positions = hist_positions[valid_row]
    hist_regrets = hist_regrets[valid_row]
    if len(hist_positions) == 0:
        return scores

    lags = current_pos - hist_positions
    w_time = time_decay_weights(lags.astype(float), cfg.time_half_life)

    if mode == "vanilla":
        w = w_time
    elif mode == "regime":
        rt = regime_z[current_pos]
        rs = regime_z[hist_positions]
        dist2 = np.sum((rs - rt) ** 2, axis=1)
        w_reg = np.exp(-dist2 / max(cfg.regime_kappa ** 2, 1e-8))
        w = w_time * w_reg
    else:
        raise ValueError(f"Unknown mode: {mode}")

    w = np.asarray(w, dtype=float)
    if np.nansum(w) <= 0 or not np.isfinite(w).any():
        return scores

    denom = np.nansum(w)
    weighted = (w.reshape(-1, 1) * hist_regrets).sum(axis=0) / denom
    return weighted.astype(float)


def compute_stress_probability(current_z: np.ndarray, regime_cols: List[str], cfg: Config) -> float:
    if current_z is None or not np.isfinite(current_z).all():
        return 0.5

    zmap = {c: float(current_z[k]) for k, c in enumerate(regime_cols)}
    score = 0.0
    norm = 0.0

    def add(col: str, weight: float) -> None:
        nonlocal score, norm
        if col in zmap and np.isfinite(zmap[col]):
            score += weight * zmap[col]
            norm += weight ** 2

    add("log_vix", 1.00)
    add("log_vix_vxv", 0.75)
    add("rv20d", 0.85)
    add("vol_of_vol_20", 0.85)
    add("hy_oas", 0.60)
    add("term_spread", -0.50)

    if norm <= 0:
        return 0.5
    score /= np.sqrt(norm)
    return float(sigmoid((score - cfg.stress_gate_center) / max(cfg.stress_gate_scale, 1e-6)))


def compute_shrunk_local_tau(
    current_pos: int,
    prior_positions: np.ndarray,
    true_scores: np.ndarray,
    regime_z: np.ndarray,
    cfg: Config,
) -> Tuple[float, float, float, float, float]:
    use_pos = np.asarray(prior_positions[-cfg.set_calibration_window:], dtype=int)
    if len(use_pos) == 0:
        return np.nan, np.nan, np.nan, 0.0, 0.0

    vals = true_scores[use_pos]
    valid = np.isfinite(vals)
    use_pos = use_pos[valid]
    vals = vals[valid]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, 0.0, 0.0

    q = min(max(1.0 - cfg.alpha, 0.0), 1.0)
    tau_global = float(np.quantile(vals, q))
    tau_local = np.nan
    lambda_local = 0.0
    n_eff_local = 0.0
    tau_final = tau_global

    if current_pos < len(regime_z) and np.isfinite(regime_z[current_pos]).all():
        hist_z = regime_z[use_pos]
        good = np.isfinite(hist_z).all(axis=1)
        hist_z = hist_z[good]
        vals_local = vals[good]
        pos_local = use_pos[good]
        if len(vals_local) > 0:
            lags = current_pos - pos_local
            w_time = time_decay_weights(lags.astype(float), cfg.time_half_life)
            dist2 = np.sum((hist_z - regime_z[current_pos]) ** 2, axis=1)
            w_reg = np.exp(-dist2 / max(cfg.local_tau_regime_kappa ** 2, 1e-8))
            weights = w_time * w_reg
            weights = np.asarray(weights, dtype=float)

            mask = np.isfinite(weights) & (weights > 0)
            weights = weights[mask]
            vals_local = vals_local[mask]

            if len(vals_local) > 0 and np.sum(weights) > 0:
                tau_local = weighted_quantile(vals_local, q, weights)
                n_eff_local = effective_sample_size(weights)
                raw_lambda = n_eff_local / max(n_eff_local + cfg.local_tau_shrinkage, 1e-8)
                if n_eff_local < cfg.min_local_tau_eff_n:
                    raw_lambda *= n_eff_local / max(cfg.min_local_tau_eff_n, 1e-8)
                lambda_local = float(np.clip(raw_lambda, 0.0, cfg.max_local_tau_weight))
                if np.isfinite(tau_local):
                    tau_final = (1.0 - lambda_local) * tau_global + lambda_local * tau_local

    return float(tau_final), float(tau_global), float(tau_local), float(lambda_local), float(n_eff_local)


def choose_model_set(
    scores: np.ndarray,
    model_names: List[str],
    tau: float,
    top_m_cap: int,
) -> List[str]:
    ranked = [(model_names[i], float(scores[i])) for i in range(len(model_names)) if np.isfinite(scores[i])]
    if len(ranked) == 0:
        return []
    ranked.sort(key=lambda x: x[1])

    if np.isfinite(tau):
        keep = [name for name, score in ranked if score <= tau]
    else:
        keep = []

    if len(keep) == 0:
        keep = [ranked[0][0]]

    top_m_cap = max(int(top_m_cap), 1)
    if len(keep) > top_m_cap:
        keep = keep[:top_m_cap]
    return keep


def rank_model_names(scores: np.ndarray, model_names: List[str]) -> List[str]:
    ranked = [(model_names[i], float(scores[i])) for i in range(len(model_names)) if np.isfinite(scores[i])]
    ranked.sort(key=lambda x: x[1])
    return [name for name, _ in ranked]


def select_specialist_names(
    ranked_names: List[str],
    selected_set: List[str],
    specialist_pool: Sequence[str],
    max_n: int,
) -> List[str]:
    specialist_pool = list(specialist_pool)
    selected_first = [m for m in ranked_names if (m in selected_set) and (m in specialist_pool)]
    if len(selected_first) == 0:
        selected_first = [m for m in ranked_names if m in specialist_pool]
    return selected_first[:max(int(max_n), 1)]


def aggregate_names(
    names: List[str],
    forecast_dict: Dict[str, float],
    mode: str,
    cfg: Config,
) -> float:
    vals = np.array([forecast_dict[m] for m in names if m in forecast_dict and np.isfinite(forecast_dict[m])], dtype=float)
    if len(vals) == 0:
        return np.nan
    if mode == "calm":
        return float(np.median(vals))
    if mode == "stress":
        vals_w = winsorize_array(vals, cfg.winsor_lower_q, cfg.winsor_upper_q)
        return float(np.quantile(vals_w, cfg.upper_quantile_q))
    raise ValueError(f"Unknown aggregation mode: {mode}")


def compute_regime_conditional_aggregates(
    ranked_names: List[str],
    selected_set: List[str],
    forecast_dict: Dict[str, float],
    p_stress: float,
    cfg: Config,
) -> Tuple[float, float, float, List[str], List[str]]:
    calm_pool = ["gru", "har", "xgb"]
    stress_pool = ["garch_t", "figarch", "har"]

    calm_names = select_specialist_names(ranked_names, selected_set, calm_pool, cfg.calm_top_m_cap)
    stress_names = select_specialist_names(ranked_names, selected_set, stress_pool, cfg.stress_top_m_cap)

    calm_pred = aggregate_names(calm_names, forecast_dict, mode="calm", cfg=cfg)
    stress_pred = aggregate_names(stress_names, forecast_dict, mode="stress", cfg=cfg)

    if np.isfinite(calm_pred) and np.isfinite(stress_pred):
        combo = float((1.0 - p_stress) * calm_pred + p_stress * stress_pred)
    elif np.isfinite(calm_pred):
        combo = float(calm_pred)
    elif np.isfinite(stress_pred):
        combo = float(stress_pred)
    else:
        combo = np.nan

    return calm_pred, stress_pred, combo, calm_names, stress_names


# =========================================================
# Metrics
# =========================================================
def summarize_forecast_metrics(
    y_var: np.ndarray,
    pred_var: np.ndarray,
    pred_name: str,
    regime_labels: np.ndarray,
    method_type: str,
    selected_sets: Optional[List[List[str]]] = None,
    best_model_names: Optional[np.ndarray] = None,
    selected_regret: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    rows = []
    mask_base = np.isfinite(y_var) & np.isfinite(pred_var)
    if mask_base.sum() == 0:
        return pd.DataFrame()

    y_vol = np.sqrt(np.maximum(y_var, 0.0))
    p_vol = np.sqrt(np.maximum(pred_var, 0.0))
    top_q = np.quantile(y_vol[mask_base], 0.90)
    under_loss = underprediction_penalty(y_vol, p_vol, eps=1e-12)

    def add_row(tag: str, mask: np.ndarray) -> None:
        if mask.sum() == 0:
            return
        row = {
            "method": pred_name,
            "method_type": method_type,
            "regime": tag,
            "n_obs": int(mask.sum()),
            "qlike_var": float(np.mean(qlike_loss(y_var[mask], pred_var[mask]))),
            "rmse_vol": float(np.sqrt(mean_squared_error(y_vol[mask], p_vol[mask]))),
            "mae_vol": float(mean_absolute_error(y_vol[mask], p_vol[mask])),
            "underprediction_rate": float(np.mean(p_vol[mask] < y_vol[mask])),
            "underprediction_loss": float(np.mean(under_loss[mask])),
            "tail_underprediction_loss": float(np.mean(under_loss[mask & (y_vol >= top_q)])) if np.any(mask & (y_vol >= top_q)) else np.nan,
            "tail_qlike_var": float(np.mean(qlike_loss(y_var[mask & (y_vol >= top_q)], pred_var[mask & (y_vol >= top_q)]))) if np.any(mask & (y_vol >= top_q)) else np.nan,
        }
        if selected_sets is not None and best_model_names is not None:
            set_sizes = np.array([len(selected_sets[i]) if selected_sets[i] is not None else np.nan for i in range(len(selected_sets))], dtype=float)
            miss_best = np.array(
                [
                    np.nan if selected_sets[i] is None or best_model_names[i] is None
                    else float(best_model_names[i] not in selected_sets[i])
                    for i in range(len(selected_sets))
                ],
                dtype=float,
            )
            turnover = np.full(len(selected_sets), np.nan)
            for i in range(1, len(selected_sets)):
                if selected_sets[i] is None or selected_sets[i - 1] is None:
                    continue
                turnover[i] = jaccard_distance(selected_sets[i - 1], selected_sets[i])
            row["avg_set_size"] = safe_mean(set_sizes[mask])
            row["miss_best_rate"] = safe_mean(miss_best[mask])
            row["avg_set_turnover"] = safe_mean(turnover[mask])
            if selected_regret is not None:
                row["avg_selected_regret"] = safe_mean(selected_regret[mask])
        rows.append(row)

    overall_mask = mask_base.copy()
    add_row("all", overall_mask)
    add_row("low", overall_mask & (regime_labels == 0))
    add_row("mid", overall_mask & (regime_labels == 1))
    add_row("high", overall_mask & (regime_labels == 2))
    return pd.DataFrame(rows)


def compute_routing_diagnostics(
    eval_daily: pd.DataFrame,
    regime_labels: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      compact_df:
        low/mid/high/all with four core diagnostics, where the two usage
        frequencies measure whether the corresponding branch actually uses
        at least one model that came from the routing set (not just fallback).
      detailed_df:
        low/mid/high/all with day-level specialist usage frequencies.
    """
    rows_compact = []
    rows_detailed = []

    regime_map = {-1: "all", 0: "low", 1: "mid", 2: "high"}

    def parse_members(s: str) -> List[str]:
        if not isinstance(s, str) or len(s) == 0:
            return []
        return [x.strip() for x in s.split(";") if x.strip()]

    def contains_member(s: str, member: str) -> float:
        return float(member in parse_members(s))

    def branch_uses_routing_set(branch_members_str: str, regime_set_str: str) -> float:
        """
        1 iff the branch contains at least one model that actually comes from
        the selected routing set on that day.
        This excludes pure fallback-only branch fills.
        """
        branch_members = set(parse_members(branch_members_str))
        selected_set = set(parse_members(regime_set_str))
        if len(branch_members) == 0 or len(selected_set) == 0:
            return 0.0
        return float(len(branch_members & selected_set) > 0)

    for regime_code, regime_tag in regime_map.items():
        if regime_code == -1:
            mask = np.ones(len(eval_daily), dtype=bool)
        else:
            mask = (regime_labels == regime_code)

        if np.sum(mask) == 0:
            continue

        sub = eval_daily.loc[mask].copy()
        n_obs = int(len(sub))

        regime_sets = sub["regime_set"].apply(parse_members).tolist()
        best_models = sub["best_model"].values
        miss_best = np.array(
            [
                np.nan if (best_models[i] is None or len(regime_sets[i]) == 0)
                else float(str(best_models[i]) not in regime_sets[i])
                for i in range(len(regime_sets))
            ],
            dtype=float,
        )

        # New compact usage-frequency definitions:
        # whether the branch actually uses at least one model from the routing set
        calm_from_routing = sub.apply(
            lambda row: branch_uses_routing_set(row["calm_members"], row["regime_set"]),
            axis=1
        ).values.astype(float)

        stress_from_routing = sub.apply(
            lambda row: branch_uses_routing_set(row["stress_members"], row["regime_set"]),
            axis=1
        ).values.astype(float)

        compact_row = {
            "regime": regime_tag,
            "n_obs": n_obs,
            "calm_selected_from_routing_set_freq": safe_mean(calm_from_routing),
            "stress_selected_from_routing_set_freq": safe_mean(stress_from_routing),
            "avg_selected_regret": safe_mean(sub["regime_selected_regret"].values.astype(float)),
            "miss_best_rate": safe_mean(miss_best),
            "interpretation": (
                "These usage frequencies measure whether the calm/stress branch "
                "actually inherits at least one model from the routing set, "
                "rather than being populated only by fallback specialists. "
                "Nonzero miss_best_rate with low selected regret means routing "
                "filters to a competitive set rather than perfectly identifying "
                "the ex post daily winner."
            ),
        }
        rows_compact.append(compact_row)

        detailed_row = {
            "regime": regime_tag,
            "n_obs": n_obs,
            "avg_set_size": safe_mean(sub["regime_set_size"].values.astype(float)),
            "avg_selected_regret": safe_mean(sub["regime_selected_regret"].values.astype(float)),
            "miss_best_rate": safe_mean(miss_best),
            # day-level usage frequency, not member-share
            "calm_freq_gru": safe_mean(sub["calm_members"].apply(lambda s: contains_member(s, "gru")).values),
            "calm_freq_har": safe_mean(sub["calm_members"].apply(lambda s: contains_member(s, "har")).values),
            "calm_freq_xgb": safe_mean(sub["calm_members"].apply(lambda s: contains_member(s, "xgb")).values),
            "stress_freq_garch_t": safe_mean(sub["stress_members"].apply(lambda s: contains_member(s, "garch_t")).values),
            "stress_freq_figarch": safe_mean(sub["stress_members"].apply(lambda s: contains_member(s, "figarch")).values),
            "stress_freq_har": safe_mean(sub["stress_members"].apply(lambda s: contains_member(s, "har")).values),
        }
        rows_detailed.append(detailed_row)

    return pd.DataFrame(rows_compact), pd.DataFrame(rows_detailed)


# =========================================================
# Main backtest
# =========================================================
def run_backtest(cfg: Config) -> Dict[str, object]:
    set_seed(cfg.seed)
    out_dir = Path(cfg.base_dir) / cfg.out_dir_name
    ensure_dir(out_dir)

    stored_model_names, active_model_names = get_model_pools(cfg)
    all_model_to_idx = {m: i for i, m in enumerate(stored_model_names)}
    active_indices = np.array([all_model_to_idx[m] for m in active_model_names], dtype=int)

    if cfg.verbose:
        print("=" * 90)
        print("Regime-Aware Online Model Prediction Set Backtest — Final Corrected Version")
        print(f"base_dir        = {cfg.base_dir}")
        print(f"symbol          = {cfg.symbol}")
        print(f"HAS_ARCH        = {HAS_ARCH}, HAS_XGB = {HAS_XGB}, HAS_TORCH = {HAS_TORCH}")
        print(f"EGARCH mode     = {cfg.egarch_mode}")
        print(f"Active pool     = {active_model_names}")
        print(f"Caps calm/stress= {cfg.calm_top_m_cap}/{cfg.stress_top_m_cap}")
        print(f"Blend rho/kappa = {cfg.low_blend_rho:.3f}/{cfg.high_blend_kappa:.3f}")
        print(f"VIX-switch thr  = {cfg.vix_switch_threshold:.2f}")
        print("=" * 90)

    panel, macro = load_inputs(cfg)
    df = prepare_symbol_dataset(panel, macro, cfg)
    tabular_features, har_features, seq_features = make_feature_lists(df)
    regime_raw, regime_cols = build_regime_matrix(df)

    valid_mask = np.isfinite(df["y_next_var"].values)
    valid_mask &= np.isfinite(regime_raw).all(axis=1)
    valid_idx = np.where(valid_mask)[0]

    warmup = max(cfg.min_train_history, 22, cfg.seq_len + 5)
    n = len(df)
    n_models = len(stored_model_names)

    pred_matrix = np.full((n, n_models), np.nan)
    loss_matrix = np.full((n, n_models), np.nan)
    regret_matrix = np.full((n, n_models), np.nan)

    best_model_name = np.array([None] * n, dtype=object)
    best_model_all = np.array([None] * n, dtype=object)

    method_cols = [
        "pred_static_best_var",
        "pred_rolling_best_var",
        "pred_vix_switch_var",
        "pred_regime_set_calm_var",
        "pred_regime_set_stress_var",
        "pred_regime_set_combo_var",
        "pred_regime_low_branch_var",
        "pred_regime_high_branch_var",
        "pred_regime_overlay_var",
    ]
    method_store = {c: np.full(n, np.nan) for c in method_cols}

    regime_set_list: List[Optional[List[str]]] = [None] * n
    regime_selected_regret = np.full(n, np.nan)

    regime_tau = np.full(n, np.nan)
    regime_tau_global = np.full(n, np.nan)
    regime_tau_local = np.full(n, np.nan)
    regime_tau_lambda = np.full(n, np.nan)
    regime_tau_neff = np.full(n, np.nan)

    stress_prob_store = np.full(n, np.nan)
    current_top_m_store = np.full(n, np.nan)
    blend_omega_store = np.full(n, np.nan)
    regime_dispersion_store = np.full(n, np.nan)
    har_floor_active_store = np.full(n, np.nan)

    calm_members_store: List[str] = [""] * n
    stress_members_store: List[str] = [""] * n
    vix_switch_state_store = np.full(n, np.nan)

    regime_true_scores = np.full(n, np.nan)
    regime_score_store = np.full((n, len(active_model_names)), np.nan)

    har_model = HARModel()
    xgb_model = XGBModelWrapper(cfg)
    gru_model = GRUModelWrapper(cfg)

    static_best_model_name: Optional[str] = None
    eval_positions = [i for i in valid_idx if i >= warmup and i < n - 1]
    if len(eval_positions) == 0:
        raise ValueError("No evaluation positions after warmup.")

    realized_eval_positions: List[int] = []

    for step, i in enumerate(tqdm(eval_positions, desc=f"Backtest {cfg.symbol}")):
        train_start = max(0, i - cfg.train_window)
        train_df = df.iloc[train_start:i].copy()
        hist_up_to_t = df.iloc[:i + 1].copy()

        recent_y = train_df["y_next_var"].dropna().tail(22)
        if len(recent_y) > 0:
            fallback_var = float(recent_y.mean())
        else:
            fallback_var = float(df["rv_var"].iloc[:i + 1].dropna().tail(22).mean())
        if not np.isfinite(fallback_var):
            fallback_var = float(np.nanmedian(df["rv_var"].dropna()))

        if (step % cfg.refit_every == 0) or (step == 0):
            har_model.fit(train_df, har_features, "y_next_var")
            xgb_model.fit(train_df, tabular_features, "y_next_var")
            gru_model.fit(train_df, seq_features, "y_next_var")

        row = df.iloc[i]
        pred_har = har_model.predict_row(row, fallback=fallback_var)
        pred_xgb = xgb_model.predict_row(row, fallback=fallback_var)
        pred_gru = gru_model.predict_row(hist_up_to_t, fallback=fallback_var)

        ret_train = df["ret"].iloc[max(0, i - cfg.train_window + 1):i + 1].values
        pred_garch = fit_predict_arch_variant(ret_train, "garch_t", fallback_var)
        pred_figarch = fit_predict_arch_variant(ret_train, "figarch", fallback_var)

        preds_today = {
            "har": pred_har,
            "garch_t": pred_garch,
            "figarch": pred_figarch,
            "gru": pred_gru,
            "xgb": pred_xgb,
        }
        if cfg.egarch_mode == "active":
            preds_today["egarch"] = fit_predict_arch_variant(ret_train, "egarch", fallback_var)

        for m, pred in preds_today.items():
            pred_matrix[i, all_model_to_idx[m]] = pred

        # ---- Naive VIX-switch baseline ----
        vix_raw_today = pd.to_numeric(df.at[i, "vix_raw"], errors="coerce") if "vix_raw" in df.columns else np.nan
        if np.isfinite(vix_raw_today):
            if vix_raw_today > cfg.vix_switch_threshold:
                method_store["pred_vix_switch_var"][i] = pred_garch if np.isfinite(pred_garch) else pred_gru
                vix_switch_state_store[i] = 1.0
            else:
                method_store["pred_vix_switch_var"][i] = pred_gru if np.isfinite(pred_gru) else pred_garch
                vix_switch_state_store[i] = 0.0
        else:
            method_store["pred_vix_switch_var"][i] = pred_gru if np.isfinite(pred_gru) else fallback_var
            vix_switch_state_store[i] = 0.0

        y_true = df.at[i, "y_next_var"]
        if np.isfinite(y_true):
            for m in stored_model_names:
                j = all_model_to_idx[m]
                if np.isfinite(pred_matrix[i, j]):
                    loss_matrix[i, j] = float(
                        composite_loss(
                            np.array([y_true]),
                            np.array([pred_matrix[i, j]]),
                            cfg.lambda_under,
                            cfg.eps,
                        )[0]
                    )

            active_losses_today = loss_matrix[i, active_indices]
            if np.isfinite(active_losses_today).any():
                active_best_local = int(np.nanargmin(active_losses_today))
                active_best_idx = int(active_indices[active_best_local])
                best_model_name[i] = stored_model_names[active_best_idx]
                active_min_loss = float(np.nanmin(active_losses_today))
                for j in range(n_models):
                    if np.isfinite(loss_matrix[i, j]):
                        regret_matrix[i, j] = max(float(loss_matrix[i, j] - active_min_loss), 0.0)

            all_losses_today = loss_matrix[i, :]
            if np.isfinite(all_losses_today).any():
                all_best_idx = int(np.nanargmin(all_losses_today))
                best_model_all[i] = stored_model_names[all_best_idx]

        prior_eval = np.array([p for p in realized_eval_positions if p < i], dtype=int)

        current_forecasts_active = {
            m: pred_matrix[i, all_model_to_idx[m]]
            for m in active_model_names
            if np.isfinite(pred_matrix[i, all_model_to_idx[m]])
        }

        if len(prior_eval) >= 20:
            recent = prior_eval[-cfg.compare_window:]
            mean_recent_loss_active = np.nanmean(loss_matrix[recent][:, active_indices], axis=0)
            if np.isfinite(mean_recent_loss_active).any():
                rb_local = int(np.nanargmin(mean_recent_loss_active))
                rb_idx = int(active_indices[rb_local])
                method_store["pred_rolling_best_var"][i] = pred_matrix[i, rb_idx]

        if static_best_model_name is None and len(prior_eval) >= cfg.static_selection_window:
            init_pos = prior_eval[:cfg.static_selection_window]
            init_mean_loss_active = np.nanmean(loss_matrix[init_pos][:, active_indices], axis=0)
            if np.isfinite(init_mean_loss_active).any():
                sb_local = int(np.nanargmin(init_mean_loss_active))
                static_best_model_name = active_model_names[sb_local]

        if static_best_model_name is not None:
            static_j = all_model_to_idx[static_best_model_name]
            method_store["pred_static_best_var"][i] = pred_matrix[i, static_j]

        enough_history_for_set = len(prior_eval) >= cfg.min_eval_history_for_set
        if enough_history_for_set:
            reg_hist = regime_raw[prior_eval]
            good_reg = np.isfinite(reg_hist).all(axis=1)
            reg_hist = reg_hist[good_reg]
            reg_positions = prior_eval[good_reg]

            if len(reg_positions) >= 30:
                scaler = StandardScaler()
                scaler.fit(reg_hist)
                regime_z = np.full_like(regime_raw, np.nan, dtype=float)
                good_all = np.isfinite(regime_raw).all(axis=1)
                regime_z[good_all] = scaler.transform(regime_raw[good_all])
            else:
                regime_z = np.full_like(regime_raw, np.nan, dtype=float)

            regret_matrix_active = regret_matrix[:, active_indices]

            if np.isfinite(regime_z[i]).all():
                cur_scores_r = compute_model_scores(i, prior_eval, regret_matrix_active, regime_z, cfg, mode="regime")
                current_z = regime_z[i]
            else:
                zero_regime = np.zeros_like(regime_raw)
                cur_scores_r = compute_model_scores(i, prior_eval, regret_matrix_active, zero_regime, cfg, mode="vanilla")
                current_z = None

            regime_score_store[i, :] = cur_scores_r
            p_stress = compute_stress_probability(current_z, regime_cols, cfg)
            stress_prob_store[i] = p_stress
            current_top_m = cfg.stress_top_m_cap if p_stress >= cfg.stress_prob_threshold else cfg.calm_top_m_cap
            current_top_m_store[i] = current_top_m

            tau_r, tau_global_r, tau_local_r, tau_lambda_r, tau_neff_r = compute_shrunk_local_tau(
                i, prior_eval, regime_true_scores, regime_z, cfg
            )
            regime_tau[i] = tau_r
            regime_tau_global[i] = tau_global_r
            regime_tau_local[i] = tau_local_r
            regime_tau_lambda[i] = tau_lambda_r
            regime_tau_neff[i] = tau_neff_r

            ranked_names = rank_model_names(cur_scores_r, active_model_names)
            set_r = choose_model_set(cur_scores_r, active_model_names, tau_r, current_top_m)
            set_r = [m for m in set_r if m in current_forecasts_active]
            if len(set_r) == 0:
                set_r = [m for m in ranked_names if m in current_forecasts_active][:1]

            regime_set_list[i] = set_r

            calm_pred, stress_pred, combo_pred, calm_names, stress_names = compute_regime_conditional_aggregates(
                ranked_names=ranked_names,
                selected_set=set_r,
                forecast_dict=current_forecasts_active,
                p_stress=p_stress,
                cfg=cfg,
            )

            method_store["pred_regime_set_calm_var"][i] = calm_pred
            method_store["pred_regime_set_stress_var"][i] = stress_pred
            method_store["pred_regime_set_combo_var"][i] = combo_pred

            calm_members_store[i] = ";".join(calm_names)
            stress_members_store[i] = ";".join(stress_names)

            set_vals = np.array([current_forecasts_active[m] for m in set_r if m in current_forecasts_active], dtype=float)
            dispersion = robust_relative_dispersion(set_vals, eps=cfg.eps)
            regime_dispersion_store[i] = dispersion

            rolling_pred = method_store["pred_rolling_best_var"][i]
            if not np.isfinite(rolling_pred):
                rolling_pred = method_store["pred_static_best_var"][i]
            if not np.isfinite(rolling_pred):
                rolling_pred = pred_har

            if np.isfinite(calm_pred):
                low_branch = (1.0 - cfg.low_blend_rho) * float(rolling_pred) + cfg.low_blend_rho * float(calm_pred)
            else:
                low_branch = float(rolling_pred)

            combo_base = combo_pred if np.isfinite(combo_pred) else rolling_pred
            stress_target = stress_pred if np.isfinite(stress_pred) else combo_base

            if np.isfinite(stress_target):
                high_branch = (1.0 - cfg.high_blend_kappa) * float(combo_base) + cfg.high_blend_kappa * float(stress_target)
            else:
                high_branch = float(combo_base)

            omega = float(sigmoid((p_stress - cfg.blend_stress_midpoint) / max(cfg.blend_stress_scale, 1e-6)))
            blend_omega_store[i] = omega

            final_pred = (1.0 - omega) * float(low_branch) + omega * float(high_branch)

            har_floor_active = float(
                (p_stress >= cfg.har_floor_stress_threshold) or
                (dispersion >= cfg.har_floor_dispersion_threshold)
            )
            har_floor_active_store[i] = har_floor_active

            if har_floor_active > 0.5 and np.isfinite(pred_har):
                final_pred = max(float(final_pred), float(pred_har))

            method_store["pred_regime_low_branch_var"][i] = float(low_branch)
            method_store["pred_regime_high_branch_var"][i] = float(high_branch)
            method_store["pred_regime_overlay_var"][i] = float(final_pred)

        if best_model_name[i] is not None:
            active_best_name_i = str(best_model_name[i])
            active_best_local = active_model_names.index(active_best_name_i)
            if np.isfinite(regime_score_store[i, active_best_local]):
                regime_true_scores[i] = regime_score_store[i, active_best_local]
            if regime_set_list[i] is not None and len(regime_set_list[i]) > 0:
                set_idxs = [all_model_to_idx[m] for m in regime_set_list[i]]
                regime_selected_regret[i] = float(np.nanmin(regret_matrix[i, set_idxs]))
            realized_eval_positions.append(i)

    # =========================================================
    # Build daily output table
    # =========================================================
    daily = df[["date", "symbol", "ret", "rv_var", "rv_vol", "y_next_var", "y_next_vol", "vix_raw"]].copy()

    for m in stored_model_names:
        j = all_model_to_idx[m]
        daily[f"pred_{m}_var"] = pred_matrix[:, j]
        daily[f"loss_{m}"] = loss_matrix[:, j]
        daily[f"regret_{m}"] = regret_matrix[:, j]

    for c in method_cols:
        daily[c] = method_store[c]

    daily["best_model"] = best_model_name
    daily["best_model_all"] = best_model_all
    daily["regime_set"] = [";".join(x) if isinstance(x, list) else "" for x in regime_set_list]
    daily["regime_set_size"] = [len(x) if isinstance(x, list) else np.nan for x in regime_set_list]

    daily["regime_tau"] = regime_tau
    daily["regime_tau_global"] = regime_tau_global
    daily["regime_tau_local"] = regime_tau_local
    daily["regime_tau_lambda"] = regime_tau_lambda
    daily["regime_tau_neff"] = regime_tau_neff
    daily["regime_selected_regret"] = regime_selected_regret

    daily["stress_prob"] = stress_prob_store
    daily["current_top_m_cap"] = current_top_m_store
    daily["blend_omega"] = blend_omega_store
    daily["har_floor_active"] = har_floor_active_store
    daily["regime_dispersion"] = regime_dispersion_store
    daily["calm_members"] = calm_members_store
    daily["stress_members"] = stress_members_store
    daily["vix_switch_state"] = vix_switch_state_store

    eval_mask = np.isfinite(daily["y_next_var"].values)
    eval_mask &= np.isfinite(daily["pred_rolling_best_var"].values)
    eval_idx = np.where(eval_mask)[0]
    if len(eval_idx) == 0:
        raise ValueError("No valid evaluation rows after rolling_best becomes available.")

    eval_daily = daily.iloc[eval_idx].copy().reset_index(drop=True)

    y_eval_vol = eval_daily["y_next_vol"].values
    q1, q2 = np.quantile(y_eval_vol[np.isfinite(y_eval_vol)], [1 / 3, 2 / 3])
    regime_labels = np.zeros(len(eval_daily), dtype=int)
    regime_labels[(y_eval_vol > q1) & (y_eval_vol <= q2)] = 1
    regime_labels[y_eval_vol > q2] = 2

    eval_daily["regime_bucket"] = regime_labels
    eval_daily["regime_name"] = [regime_name_map(x) for x in regime_labels]

    # =========================================================
    # Summary metrics
    # =========================================================
    summary_frames = []

    for m in stored_model_names:
        pred_col = f"pred_{m}_var"
        summary_frames.append(
            summarize_forecast_metrics(
                y_var=eval_daily["y_next_var"].values,
                pred_var=eval_daily[pred_col].values,
                pred_name=m,
                regime_labels=regime_labels,
                method_type="single_model",
            )
        )

    baseline_map = {
        "static_best": ("pred_static_best_var", "baseline", False),
        "rolling_best": ("pred_rolling_best_var", "baseline", False),
        "vix_switch": ("pred_vix_switch_var", "naive_baseline", False),
        "regime_set_calm": ("pred_regime_set_calm_var", "model_set", True),
        "regime_set_stress": ("pred_regime_set_stress_var", "model_set", True),
        "regime_set_combo": ("pred_regime_set_combo_var", "model_set", True),
        "regime_low_branch": ("pred_regime_low_branch_var", "blend_component", False),
        "regime_high_branch": ("pred_regime_high_branch_var", "blend_component", False),
        "regime_overlay": ("pred_regime_overlay_var", "overlay_policy", False),
    }

    selected_sets_eval = eval_daily["regime_set"].apply(
        lambda s: s.split(";") if isinstance(s, str) and len(s) > 0 else []
    ).tolist()
    selected_regret_eval = eval_daily["regime_selected_regret"].values
    best_model_eval = eval_daily["best_model"].values

    for name, (col, mtype, use_set_meta) in baseline_map.items():
        summary_frames.append(
            summarize_forecast_metrics(
                y_var=eval_daily["y_next_var"].values,
                pred_var=eval_daily[col].values,
                pred_name=name,
                regime_labels=regime_labels,
                method_type=mtype,
                selected_sets=selected_sets_eval if use_set_meta else None,
                best_model_names=best_model_eval if use_set_meta else None,
                selected_regret=selected_regret_eval if use_set_meta else None,
            )
        )

    summary_df = pd.concat(summary_frames, axis=0, ignore_index=True)
    model_level_metrics = summary_df[
        (summary_df["method_type"] == "single_model") & (summary_df["regime"] == "all")
    ].copy()

    # =========================================================
    # Routing diagnostics
    # =========================================================
    routing_compact_df, routing_detailed_df = compute_routing_diagnostics(
        eval_daily=eval_daily,
        regime_labels=regime_labels,
    )

    # =========================================================
    # Asset-level DM tests
    # =========================================================
    dm_rb = diebold_mariano_test(
        y=eval_daily["y_next_var"].values,
        pred_a=eval_daily["pred_regime_overlay_var"].values,
        pred_b=eval_daily["pred_rolling_best_var"].values,
        loss_name="qlike",
        eps=cfg.eps,
    )
    dm_vs = diebold_mariano_test(
        y=eval_daily["y_next_var"].values,
        pred_a=eval_daily["pred_regime_overlay_var"].values,
        pred_b=eval_daily["pred_vix_switch_var"].values,
        loss_name="qlike",
        eps=cfg.eps,
    )
    dm_df = pd.DataFrame(
        [
            {"asset": cfg.symbol, "method_a": "regime_overlay", "method_b": "rolling_best", "loss_name": "qlike_var", **dm_rb},
            {"asset": cfg.symbol, "method_a": "regime_overlay", "method_b": "vix_switch", "loss_name": "qlike_var", **dm_vs},
        ]
    )

    # =========================================================
    # Save
    # =========================================================
    daily_out = out_dir / "daily_outputs.csv"
    summary_out = out_dir / "summary_metrics.csv"
    model_out = out_dir / "model_level_metrics.csv"
    routing_compact_out = out_dir / "routing_diagnostics_compact.csv"
    routing_detailed_out = out_dir / "routing_diagnostics_detailed.csv"
    dm_out = out_dir / "dm_test_results.csv"
    config_out = out_dir / "config_used.json"

    daily.to_csv(daily_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    model_level_metrics.to_csv(model_out, index=False)
    routing_compact_df.to_csv(routing_compact_out, index=False)
    routing_detailed_df.to_csv(routing_detailed_out, index=False)
    dm_df.to_csv(dm_out, index=False)

    run_info = {
        "symbol": cfg.symbol,
        "n_total_rows": int(len(df)),
        "n_eval_rows": int(len(eval_daily)),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "eval_start": str(eval_daily["date"].min().date()),
        "eval_end": str(eval_daily["date"].max().date()),
        "has_arch": HAS_ARCH,
        "has_xgb": HAS_XGB,
        "has_torch": HAS_TORCH,
        "regime_cols": regime_cols,
        "stored_model_pool": stored_model_names,
        "active_model_pool": active_model_names,
        "egarch_mode": cfg.egarch_mode,
        "calm_top_m_cap": cfg.calm_top_m_cap,
        "stress_top_m_cap": cfg.stress_top_m_cap,
        "stress_prob_threshold": cfg.stress_prob_threshold,
        "vix_switch_threshold": cfg.vix_switch_threshold,
        "loss_definition": "QLIKE(next-day realized variance) + lambda_under * UnderPenalty",
        "target_definition": "y_next_var = (gk_proxy_{t+1})^2",
        "dm_test_overlay_vs_rolling_best": {
            "dm_statistic": dm_rb["dm_stat"],
            "p_value": dm_rb["p_value"],
            "favors_overlay": dm_rb["favors_a"],
            "loss_fn": "QLIKE",
            "interpretation": "negative statistic means overlay has lower loss",
        },
        "dm_test_overlay_vs_vix_switch": {
            "dm_statistic": dm_vs["dm_stat"],
            "p_value": dm_vs["p_value"],
            "favors_overlay": dm_vs["favors_a"],
            "loss_fn": "QLIKE",
            "interpretation": "negative statistic means overlay has lower loss",
        },
        "aggregation_definition": {
            "calm_pool": ["gru", "har", "xgb"],
            "stress_pool": ["garch_t", "figarch", "har"],
            "calm_aggregation": "median of calm specialist subset",
            "stress_aggregation": {
                "type": "winsorized q75",
                "base_quantile": cfg.upper_quantile_q,
                "winsor_lower_q": cfg.winsor_lower_q,
                "winsor_upper_q": cfg.winsor_upper_q,
            },
        },
        "tau_definition": {
            "type": "regime-local weighted tau with shrinkage to global tau",
            "local_tau_regime_kappa": cfg.local_tau_regime_kappa,
            "local_tau_shrinkage": cfg.local_tau_shrinkage,
            "max_local_tau_weight": cfg.max_local_tau_weight,
        },
        "overlay_definition": {
            "low_branch": "y_low = (1-rho)*rolling_best + rho*calm_pred",
            "high_branch": "y_high = (1-kappa)*combo_pred + kappa*stress_pred",
            "omega_gate": "omega_t is a monotone stress-probability gate",
            "conditional_har_floor": {
                "enabled_when": "stress probability is high or specialist dispersion is high",
                "har_floor_stress_threshold": cfg.har_floor_stress_threshold,
                "har_floor_dispersion_threshold": cfg.har_floor_dispersion_threshold,
            },
        },
        "routing_diagnostics_definition": {
            "compact": "regime-level calm/stress branch usage, selected regret, and miss-best rate",
            "detailed": "day-level specialist usage frequencies within calm/stress branches",
        },
        "notes": [
            "Missing rolling_vol and ewma_vol features are now explicitly created.",
            "VIX-switch baseline uses raw VIX > threshold -> GARCH-t, else GRU.",
            "Routing diagnostics are day-level usage frequencies, not member-share composition.",
            "DM summary should count only cases where p < 0.05 and the statistic favors overlay.",
        ],
    }

    payload = asdict(cfg)
    payload["run_info"] = run_info
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # =========================================================
    # Print concise report
    # =========================================================
    overall = summary_df[summary_df["regime"] == "all"].copy()
    overall = overall.sort_values(["method_type", "qlike_var"])

    print("\n" + "=" * 90)
    print(f"Finished backtest for {cfg.symbol}")
    print(f"Evaluation rows: {len(eval_daily)}")
    print("=" * 90)

    show_cols = [
        "method", "method_type", "qlike_var", "rmse_vol", "mae_vol",
        "underprediction_rate", "underprediction_loss", "avg_set_size", "miss_best_rate"
    ]
    use_show_cols = [c for c in show_cols if c in overall.columns]
    print(overall[use_show_cols].to_string(index=False))

    if not routing_compact_df.empty:
        print("\nRouting diagnostics (compact):")
        print(routing_compact_df.to_string(index=False))

    if not dm_df.empty:
        print("\nDM tests:")
        print(dm_df.to_string(index=False))

    print("\nFiles:")
    print(f"  {daily_out}")
    print(f"  {summary_out}")
    print(f"  {model_out}")
    print(f"  {routing_compact_out}")
    print(f"  {routing_detailed_out}")
    print(f"  {dm_out}")
    print(f"  {config_out}")

    return {
        "eval_daily": eval_daily,
        "summary_df": summary_df,
        "routing_compact": routing_compact_df,
        "routing_detailed": routing_detailed_df,
        "dm_results": dm_df,
        "paths": {
            "daily_out": str(daily_out),
            "summary_out": str(summary_out),
            "model_out": str(model_out),
            "routing_compact_out": str(routing_compact_out),
            "routing_detailed_out": str(routing_detailed_out),
            "dm_out": str(dm_out),
            "config_out": str(config_out),
        },
    }


# =========================================================
# Cross-asset + ablation helpers
# =========================================================
DEFAULT_ASSET_ORDER = ["EEM", "GLD", "IWM", "QQQ", "SPY", "TLT"]


def parse_csv_arg(text: str | None) -> List[str]:
    if text is None or str(text).strip() == "":
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def safe_slug(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def discover_assets(cfg: Config) -> List[str]:
    panel_path = Path(cfg.base_dir) / cfg.panel_file
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")
    df = pd.read_csv(panel_path, usecols=["symbol"])
    symbols = sorted(pd.Series(df["symbol"].dropna().astype(str).unique()).tolist())
    preferred = [s for s in DEFAULT_ASSET_ORDER if s in symbols]
    rest = [s for s in symbols if s not in preferred]
    return preferred + rest


def build_ablation_specs() -> Dict[str, Dict[str, object]]:
    return {
        "full": {},
        "no_local_tau": {
            "max_local_tau_weight": 0.0,
        },
        "no_har_floor": {
            "har_floor_stress_threshold": 1.1,
            "har_floor_dispersion_threshold": 1e9,
        },
        "fixed_cap_2": {
            "calm_top_m_cap": 2,
            "stress_top_m_cap": 2,
        },
        "simple_state_blend": {
            "low_blend_rho": 0.0,
            "high_blend_kappa": 0.0,
            "har_floor_stress_threshold": 1.1,
            "har_floor_dispersion_threshold": 1e9,
        },
        "lighter_high_branch": {
            "high_blend_kappa": 0.20,
        },
        "heavier_high_branch": {
            "high_blend_kappa": 0.50,
        },
    }


def apply_overrides(cfg: Config, overrides: Dict[str, object]) -> Config:
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Config has no field '{k}'")
        setattr(cfg, k, v)
    return cfg


def read_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_one_batch_job(
    base_cfg: Config,
    asset: str,
    ablation_name: str,
    overrides: Dict[str, object],
    batch_root_name: str,
) -> Dict[str, object]:
    cfg = copy.deepcopy(base_cfg)
    cfg.symbol = asset
    cfg.out_dir_name = f"{safe_slug(batch_root_name)}/{safe_slug(asset)}/{safe_slug(ablation_name)}"
    apply_overrides(cfg, overrides)

    bt_result = run_backtest(cfg)

    out_dir = Path(cfg.base_dir) / cfg.out_dir_name
    summary_out = out_dir / "summary_metrics.csv"
    config_out = out_dir / "config_used.json"
    model_out = out_dir / "model_level_metrics.csv"
    daily_out = out_dir / "daily_outputs.csv"
    routing_compact_out = out_dir / "routing_diagnostics_compact.csv"
    routing_detailed_out = out_dir / "routing_diagnostics_detailed.csv"
    dm_out = out_dir / "dm_test_results.csv"

    if not summary_out.exists():
        raise FileNotFoundError(f"Expected summary file missing: {summary_out}")
    if not config_out.exists():
        raise FileNotFoundError(f"Expected config file missing: {config_out}")

    return {
        "asset": asset,
        "ablation": ablation_name,
        "out_dir": str(out_dir),
        "daily_out": str(daily_out),
        "summary_out": str(summary_out),
        "model_out": str(model_out),
        "routing_compact_out": str(routing_compact_out),
        "routing_detailed_out": str(routing_detailed_out),
        "dm_out": str(dm_out),
        "config_out": str(config_out),
        "summary_df": pd.read_csv(summary_out),
        "routing_compact_df": pd.read_csv(routing_compact_out) if routing_compact_out.exists() else pd.DataFrame(),
        "routing_detailed_df": pd.read_csv(routing_detailed_out) if routing_detailed_out.exists() else pd.DataFrame(),
        "dm_df": pd.read_csv(dm_out) if dm_out.exists() else pd.DataFrame(),
        "config_payload": read_json(config_out),
        "bt_result": bt_result,
    }


def add_relative_baseline_tables(by_regime_df: pd.DataFrame) -> pd.DataFrame:
    if by_regime_df.empty:
        return pd.DataFrame()

    needed = {"asset", "ablation", "regime", "method", "qlike_var", "underprediction_loss"}
    if not needed.issubset(set(by_regime_df.columns)):
        return pd.DataFrame()

    piv = by_regime_df.pivot_table(
        index=["asset", "ablation", "regime"],
        columns="method",
        values=["qlike_var", "underprediction_loss"],
        aggfunc="first",
    )
    piv.columns = [f"{a}__{b}" for a, b in piv.columns.to_flat_index()]
    piv = piv.reset_index()

    rows = []
    for _, row in piv.iterrows():
        for method_col in [c for c in piv.columns if c.startswith("qlike_var__")]:
            method = method_col.split("__", 1)[1]
            qlike_val = row.get(f"qlike_var__{method}")
            upl_val = row.get(f"underprediction_loss__{method}")
            if not pd.notna(qlike_val):
                continue
            out = {
                "asset": row["asset"],
                "ablation": row["ablation"],
                "regime": row["regime"],
                "method": method,
                "qlike_var": qlike_val,
                "underprediction_loss": upl_val,
            }
            for baseline in ["rolling_best", "har", "vix_switch"]:
                q_base = row.get(f"qlike_var__{baseline}")
                u_base = row.get(f"underprediction_loss__{baseline}")
                out[f"delta_qlike_vs_{baseline}"] = (qlike_val - q_base) if pd.notna(q_base) else np.nan
                out[f"delta_underprediction_loss_vs_{baseline}"] = (upl_val - u_base) if pd.notna(u_base) and pd.notna(upl_val) else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def aggregate_batch_results(records: List[Dict[str, object]], batch_out_dir: Path) -> Dict[str, pd.DataFrame]:
    manifest_rows = []
    overall_rows = []
    regime_rows = []
    routing_compact_rows = []
    routing_detailed_rows = []
    dm_rows = []

    for rec in records:
        cfg_payload = rec.get("config_payload", {})
        run_info = cfg_payload.get("run_info", {}) if isinstance(cfg_payload, dict) else {}
        manifest_rows.append({
            "asset": rec.get("asset"),
            "ablation": rec.get("ablation"),
            "status": rec.get("status", "success"),
            "error_message": rec.get("error_message", ""),
            "out_dir": rec.get("out_dir", ""),
            "summary_out": rec.get("summary_out", ""),
            "config_out": rec.get("config_out", ""),
            "n_eval_rows": run_info.get("n_eval_rows"),
            "eval_start": run_info.get("eval_start"),
            "eval_end": run_info.get("eval_end"),
        })

        if rec.get("status") != "success":
            continue

        summary_df = rec["summary_df"].copy()
        summary_df["asset"] = rec["asset"]
        summary_df["ablation"] = rec["ablation"]
        overall_rows.append(summary_df[summary_df["regime"] == "all"].copy())
        regime_rows.append(summary_df.copy())

        rc = rec.get("routing_compact_df", pd.DataFrame()).copy()
        if not rc.empty:
            rc["asset"] = rec["asset"]
            rc["ablation"] = rec["ablation"]
            routing_compact_rows.append(rc)

        rd = rec.get("routing_detailed_df", pd.DataFrame()).copy()
        if not rd.empty:
            rd["asset"] = rec["asset"]
            rd["ablation"] = rec["ablation"]
            routing_detailed_rows.append(rd)

        dm = rec.get("dm_df", pd.DataFrame()).copy()
        if not dm.empty:
            dm["ablation"] = rec["ablation"]
            dm_rows.append(dm)

    manifest_df = pd.DataFrame(manifest_rows)
    overall_df = pd.concat(overall_rows, ignore_index=True) if overall_rows else pd.DataFrame()
    by_regime_df = pd.concat(regime_rows, ignore_index=True) if regime_rows else pd.DataFrame()
    routing_compact_df = pd.concat(routing_compact_rows, ignore_index=True) if routing_compact_rows else pd.DataFrame()
    routing_detailed_df = pd.concat(routing_detailed_rows, ignore_index=True) if routing_detailed_rows else pd.DataFrame()
    dm_tests_df = pd.concat(dm_rows, ignore_index=True) if dm_rows else pd.DataFrame()

    metric_cols = [
        "qlike_var",
        "rmse_vol",
        "mae_vol",
        "underprediction_rate",
        "underprediction_loss",
        "avg_set_size",
        "miss_best_rate",
        "tail_underprediction_loss",
        "tail_qlike_var",
    ]
    agg_map = {c: ["mean", "median", "std"] for c in metric_cols if c in by_regime_df.columns}
    if not by_regime_df.empty:
        method_agg = (
            by_regime_df
            .groupby(["ablation", "method", "method_type", "regime"], dropna=False)
            .agg(agg_map)
            .reset_index()
        )
        method_agg.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in method_agg.columns.to_flat_index()
        ]
    else:
        method_agg = pd.DataFrame()

    win_rows = []
    if not by_regime_df.empty and "qlike_var" in by_regime_df.columns:
        for (ablation, regime), block in by_regime_df.groupby(["ablation", "regime"], dropna=False):
            per_asset = []
            for asset, asset_block in block.groupby("asset"):
                asset_block = asset_block.dropna(subset=["qlike_var"]).copy()
                if asset_block.empty:
                    continue
                winner = asset_block.loc[asset_block["qlike_var"].idxmin()]
                per_asset.append({
                    "asset": asset,
                    "winner_method": winner["method"],
                    "winner_method_type": winner.get("method_type"),
                })
            if per_asset:
                wins_df = pd.DataFrame(per_asset)
                wins_count = (
                    wins_df.groupby(["winner_method", "winner_method_type"], dropna=False)
                    .size()
                    .reset_index(name="n_asset_wins")
                )
                wins_count["ablation"] = ablation
                wins_count["regime"] = regime
                win_rows.append(wins_count)
    win_counts_df = pd.concat(win_rows, ignore_index=True) if win_rows else pd.DataFrame()

    relative_df = add_relative_baseline_tables(by_regime_df)
    if not relative_df.empty:
        delta_cols = [c for c in relative_df.columns if c.startswith("delta_")]
        relative_agg = (
            relative_df.groupby(["ablation", "method", "regime"], dropna=False)
            .agg({c: ["mean", "median"] for c in delta_cols})
            .reset_index()
        )
        relative_agg.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in relative_agg.columns.to_flat_index()
        ]
    else:
        relative_agg = pd.DataFrame()

    if not routing_compact_df.empty:
        routing_compact_agg = (
            routing_compact_df.groupby(["ablation", "regime"], dropna=False)
            .agg({
                "n_obs": "sum",
                "calm_selected_from_routing_set_freq": "mean",
                "stress_selected_from_routing_set_freq": "mean",
                "avg_selected_regret": "mean",
                "miss_best_rate": "mean",
            })
            .reset_index()
        )
    else:
        routing_compact_agg = pd.DataFrame()

    outputs = {
        "batch_manifest": manifest_df,
        "cross_asset_overall_summary": overall_df,
        "cross_asset_by_regime_summary": by_regime_df,
        "cross_asset_method_aggregate": method_agg,
        "cross_asset_method_win_counts": win_counts_df,
        "cross_asset_relative_to_baselines": relative_df,
        "cross_asset_relative_aggregate": relative_agg,
        "cross_asset_routing_diagnostics_compact": routing_compact_df,
        "cross_asset_routing_diagnostics_compact_aggregate": routing_compact_agg,
        "cross_asset_routing_diagnostics_detailed": routing_detailed_df,
        "cross_asset_dm_tests": dm_tests_df,
    }

    for name, df in outputs.items():
        df.to_csv(batch_out_dir / f"{name}.csv", index=False)

    sentence_rb = format_dm_summary_sentence(dm_tests_df, "regime_overlay", "rolling_best")
    sentence_vs = format_dm_summary_sentence(dm_tests_df, "regime_overlay", "vix_switch")
    with open(batch_out_dir / "cross_asset_dm_summary_sentence.txt", "w", encoding="utf-8") as f:
        if sentence_rb:
            f.write(sentence_rb + "\n")
        if sentence_vs:
            f.write(sentence_vs + "\n")

    return outputs


def run_cross_asset_ablation_batch(
    base_cfg: Config,
    assets: List[str],
    ablation_names: List[str],
    batch_out_dir_name: str,
    continue_on_error: bool = True,
) -> Dict[str, pd.DataFrame]:
    ablation_specs = build_ablation_specs()
    batch_out_dir = Path(base_cfg.base_dir) / batch_out_dir_name
    ensure_dir(batch_out_dir)

    records: List[Dict[str, object]] = []
    jobs = [(asset, ablation) for asset in assets for ablation in ablation_names]

    for asset, ablation_name in tqdm(jobs, desc="Batch jobs"):
        overrides = ablation_specs[ablation_name]
        try:
            rec = run_one_batch_job(
                base_cfg=base_cfg,
                asset=asset,
                ablation_name=ablation_name,
                overrides=overrides,
                batch_root_name=batch_out_dir_name,
            )
            rec["status"] = "success"
            rec["error_message"] = ""
        except Exception as e:
            rec = {
                "asset": asset,
                "ablation": ablation_name,
                "status": "failed",
                "error_message": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "out_dir": str(Path(base_cfg.base_dir) / batch_out_dir_name / safe_slug(asset) / safe_slug(ablation_name)),
                "summary_out": "",
                "config_out": "",
                "config_payload": {},
            }
            if not continue_on_error:
                raise
        records.append(rec)

    return aggregate_batch_results(records=records, batch_out_dir=batch_out_dir)


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regime-aware online model set backtest with corrected routing diagnostics, naive VIX-switch baseline, and cross-asset DM tests."
    )
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--panel-file", type=str, default=None)
    p.add_argument("--macro-file", type=str, default=None)
    p.add_argument("--symbol", type=str, default="SPY")
    p.add_argument("--lambda-under", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--train-window", type=int, default=None)
    p.add_argument("--compare-window", type=int, default=None)
    p.add_argument("--refit-every", type=int, default=None)
    p.add_argument("--egarch-mode", type=str, default=None, choices=["drop", "active"])
    p.add_argument("--calm-top-m-cap", type=int, default=None)
    p.add_argument("--stress-top-m-cap", type=int, default=None)
    p.add_argument("--stress-prob-threshold", type=float, default=None)
    p.add_argument("--vix-switch-threshold", type=float, default=None)
    p.add_argument("--low-blend-rho", type=float, default=None)
    p.add_argument("--high-blend-kappa", type=float, default=None)
    p.add_argument("--blend-stress-midpoint", type=float, default=None)
    p.add_argument("--blend-stress-scale", type=float, default=None)
    p.add_argument("--har-floor-stress-threshold", type=float, default=None)
    p.add_argument("--har-floor-dispersion-threshold", type=float, default=None)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--batch-mode", action="store_true", help="Force cross-asset + ablation batch mode.")
    p.add_argument("--single-asset-mode", action="store_true", help="Force single-asset mode even when launched with no CLI args.")
    p.add_argument("--assets", type=str, default="EEM,GLD,IWM,QQQ,SPY,TLT")
    p.add_argument("--ablations", type=str, default="full,no_local_tau,no_har_floor,fixed_cap_2,simple_state_blend")
    p.add_argument("--batch-out-dir-name", type=str, default="regime_overlay_cross_asset_ablation")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    launched_without_cli_args = len(__import__("sys").argv) == 1
    if launched_without_cli_args and not args.single_asset_mode:
        args.batch_mode = True

    cfg = Config()

    if args.base_dir is not None:
        cfg.base_dir = args.base_dir
    if args.panel_file is not None:
        cfg.panel_file = args.panel_file
    if args.macro_file is not None:
        cfg.macro_file = args.macro_file
    if args.symbol is not None:
        cfg.symbol = args.symbol
    if args.lambda_under is not None:
        cfg.lambda_under = float(args.lambda_under)
    if args.alpha is not None:
        cfg.alpha = float(args.alpha)
    if args.train_window is not None:
        cfg.train_window = int(args.train_window)
    if args.compare_window is not None:
        cfg.compare_window = int(args.compare_window)
    if args.refit_every is not None:
        cfg.refit_every = int(args.refit_every)
    if args.egarch_mode is not None:
        cfg.egarch_mode = str(args.egarch_mode)
    if args.calm_top_m_cap is not None:
        cfg.calm_top_m_cap = int(args.calm_top_m_cap)
    if args.stress_top_m_cap is not None:
        cfg.stress_top_m_cap = int(args.stress_top_m_cap)
    if args.stress_prob_threshold is not None:
        cfg.stress_prob_threshold = float(args.stress_prob_threshold)
    if args.vix_switch_threshold is not None:
        cfg.vix_switch_threshold = float(args.vix_switch_threshold)
    if args.low_blend_rho is not None:
        cfg.low_blend_rho = float(args.low_blend_rho)
    if args.high_blend_kappa is not None:
        cfg.high_blend_kappa = float(args.high_blend_kappa)
    if args.blend_stress_midpoint is not None:
        cfg.blend_stress_midpoint = float(args.blend_stress_midpoint)
    if args.blend_stress_scale is not None:
        cfg.blend_stress_scale = float(args.blend_stress_scale)
    if args.har_floor_stress_threshold is not None:
        cfg.har_floor_stress_threshold = float(args.har_floor_stress_threshold)
    if args.har_floor_dispersion_threshold is not None:
        cfg.har_floor_dispersion_threshold = float(args.har_floor_dispersion_threshold)
    if args.force_cpu:
        cfg.force_cpu = True

    if not HAS_ARCH:
        warnings.warn(
            "arch package not found. GARCH-t / FIGARCH / EGARCH will fallback to recent-mean variance when requested. Install with: pip install arch"
        )
    if not HAS_TORCH:
        warnings.warn(
            "torch package not found. GRU will fallback to recent-mean variance. Install PyTorch if you want the GRU model."
        )

    if args.batch_mode:
        all_abls = build_ablation_specs()
        requested_ablations = parse_csv_arg(args.ablations)
        if len(requested_ablations) == 0:
            requested_ablations = ["full"]
        missing_abls = [a for a in requested_ablations if a not in all_abls]
        if missing_abls:
            raise ValueError(f"Unknown ablation names: {missing_abls}. Available: {sorted(all_abls.keys())}")

        requested_assets = parse_csv_arg(args.assets)
        if len(requested_assets) == 0:
            assets = discover_assets(cfg)
        else:
            available_assets = set(discover_assets(cfg))
            missing_assets = [a for a in requested_assets if a not in available_assets]
            if missing_assets:
                raise ValueError(f"Requested assets not found in panel file: {missing_assets}")
            assets = requested_assets

        print("=" * 100)
        print("Cross-Asset + Ablation Batch Mode")
        print(f"base_dir     = {cfg.base_dir}")
        print(f"assets       = {assets}")
        print(f"ablations    = {requested_ablations}")
        print(f"batch_out    = {Path(cfg.base_dir) / args.batch_out_dir_name}")
        print("=" * 100)

        outputs = run_cross_asset_ablation_batch(
            base_cfg=cfg,
            assets=assets,
            ablation_names=requested_ablations,
            batch_out_dir_name=args.batch_out_dir_name,
            continue_on_error=bool(args.continue_on_error),
        )

        method_agg = outputs.get("cross_asset_method_aggregate", pd.DataFrame())
        if not method_agg.empty:
            show_cols = [
                c for c in [
                    "ablation", "method", "method_type", "regime",
                    "qlike_var_mean", "qlike_var_median", "rmse_vol_mean",
                    "underprediction_loss_mean", "avg_set_size_mean", "miss_best_rate_mean",
                ] if c in method_agg.columns
            ]
            print("\nCross-asset aggregate snapshot:")
            print(method_agg[show_cols].sort_values(["ablation", "regime", "qlike_var_mean"]).head(60).to_string(index=False))

        rel_agg = outputs.get("cross_asset_relative_aggregate", pd.DataFrame())
        if not rel_agg.empty:
            rel_cols = [
                c for c in [
                    "ablation", "method", "regime",
                    "delta_qlike_vs_rolling_best_mean",
                    "delta_qlike_vs_har_mean",
                    "delta_qlike_vs_vix_switch_mean",
                    "delta_underprediction_loss_vs_rolling_best_mean",
                ] if c in rel_agg.columns
            ]
            print("\nCross-asset relative-to-baseline snapshot:")
            print(rel_agg[rel_cols].sort_values(["ablation", "regime", "delta_qlike_vs_rolling_best_mean"]).head(60).to_string(index=False))

        routing_compact = outputs.get("cross_asset_routing_diagnostics_compact_aggregate", pd.DataFrame())
        if not routing_compact.empty:
            print("\nCross-asset routing diagnostics (compact aggregate):")
            print(routing_compact.to_string(index=False))

        dm_df = outputs.get("cross_asset_dm_tests", pd.DataFrame())
        if not dm_df.empty:
            print("\nCross-asset DM tests:")
            print(dm_df.to_string(index=False))

            sentence_rb = format_dm_summary_sentence(dm_df, "regime_overlay", "rolling_best")
            sentence_vs = format_dm_summary_sentence(dm_df, "regime_overlay", "vix_switch")
            print("\nReady-to-use DM sentences:")
            if sentence_rb:
                print(sentence_rb)
            if sentence_vs:
                print(sentence_vs)

        batch_out = Path(cfg.base_dir) / args.batch_out_dir_name
        print("\nSaved files:")
        for name in [
            "batch_manifest.csv",
            "cross_asset_overall_summary.csv",
            "cross_asset_by_regime_summary.csv",
            "cross_asset_method_aggregate.csv",
            "cross_asset_method_win_counts.csv",
            "cross_asset_relative_to_baselines.csv",
            "cross_asset_relative_aggregate.csv",
            "cross_asset_routing_diagnostics_compact.csv",
            "cross_asset_routing_diagnostics_compact_aggregate.csv",
            "cross_asset_routing_diagnostics_detailed.csv",
            "cross_asset_dm_tests.csv",
            "cross_asset_dm_summary_sentence.txt",
        ]:
            print(f"  {batch_out / name}")

    else:
        run_backtest(cfg)


if __name__ == "__main__":
    main()
