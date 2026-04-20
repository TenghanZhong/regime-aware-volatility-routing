# Regime-Aware Volatility Routing

Code and representative outputs for a regime-aware specialist routing framework for next-day ETF volatility forecasting.

This repository implements a rolling walk-forward forecasting system that combines econometric and machine-learning specialists through risk-sensitive online scoring, state-dependent routing, and stress-aware forecast blending.

The framework is designed for next-day ETF realized-variance forecasting under changing market conditions, with particular emphasis on stressed-state robustness and protection against underprediction.

## Overview

The repository supports four main components:

- next-day ETF volatility / realized-variance forecasting
- regime-aware specialist routing across calm and stressed states
- cross-asset evaluation and summary aggregation
- focused ablation on stress-aware routing components

The implementation follows a rolling walk-forward design and reports both predictive accuracy and routing-behavior diagnostics.

## Repository Structure

- `run_regime_aware_routing.py`  
  Main walk-forward backtest for regime-aware specialist routing, including:
  - single-model forecasts
  - adaptive baselines
  - routing-set construction
  - state-dependent forecast combination
  - routing diagnostics
  - Diebold–Mariano tests
  - cross-asset batch aggregation

- `ablation_stress_components.py.py`  
  Focused ablation script for stress-aware design components.

- `Plot.py`  
  Figure-generation script for paper-ready plots and cross-asset visual summaries.

- `data/`  
  Input data directory containing:
  - `multiasset_daily_10y_panel_model.csv`
  - `master_daily_features_macro_dailyonly_raw_only.csv`

- `Result/`  
  Representative output directory currently containing:
  - `cross_asset_method_aggregate.csv`
  - `cross_asset_relative_aggregate.csv`

- `LICENSE`  
  Repository license.

## Main Modeling Idea

The framework treats next-day ETF volatility forecasting as a routing problem rather than a single-model selection problem.

At each forecast date, candidate models are evaluated online using a risk-sensitive loss. Competitive models are retained in a routing set, then routed into calm and stress specialist branches. These branches are blended into a final overlay forecast using observable market-state information and a stress-dependent gating mechanism.

The implementation supports:

1. next-day realized variance / volatility forecasting
2. candidate model pool:
   - HAR-RV
   - GARCH-t
   - FIGARCH
   - GRU
   - XGBoost
   - optional EGARCH
3. risk-sensitive model scoring using:
   - `QLIKE + lambda_under * UnderPenalty`
4. adaptive baselines:
   - static-best
   - rolling-best
   - naive VIX-switch
5. regime-aware routing-set construction
6. calm / stress specialist aggregation
7. low-state / high-state branch blending
8. conditional HAR floor
9. routing diagnostics
10. asset-level Diebold–Mariano tests with Newey–West HAC variance

## Target Construction

The forecasting target is built internally from the ETF panel.

For each date:

- `rv_var = gk_proxy^2`
- `rv_vol = sqrt(rv_var)`
- `y_next_var = rv_var shifted by -1`
- `y_next_vol = sqrt(y_next_var)`

QLIKE is evaluated on the variance scale.  
RMSE and MAE are also reported on the corresponding volatility scale.

## Feature Construction

The code constructs forecasting inputs internally from the panel and macro files.

### ETF / volatility features
- HAR-style variance inputs
- lagged variance terms
- return lags
- absolute-return lags
- rolling variance means
- rolling variance standard deviations
- log-variance features
- volatility-of-volatility features
- rolling volatility features
- EWMA volatility

### Macro-financial state features
- `log_vix`
- `log_vix_vxv`
- `d5_log_vix`
- `term_spread`
- `hy_oas`
- `vix_raw`
- `rv20d`

## Default Configuration

Important defaults in the current implementation include:

- target proxy: `gk_proxy`
- training window: `504`
- comparison window: `252`
- set calibration window: `252`
- minimum training history: `504`
- refit frequency: every `21` trading days
- static selection window: `252`
- `lambda_under = 1.0`
- score history window: `252`
- `alpha = 0.10`
- GRU sequence length: `20`
- naive VIX-switch threshold: `20.0`
- calm top-m cap: `1`
- stress top-m cap: `2`
- stress aggregation quantile: `0.75`
- winsorization bounds: `0.10` and `0.90`
- HAR floor stress threshold: `0.65`
- HAR floor dispersion threshold: `0.20`
- random seed: `42`

## Supported Methods

### Single-model forecasters
- HAR-RV
- GARCH-t
- FIGARCH
- GRU
- XGBoost

Optional:
- EGARCH

### Adaptive baselines
- static-best
- rolling-best
- naive VIX-switch

### Routing outputs
- calm branch
- stress branch
- regime-conditional combo branch
- low-state branch
- high-state branch
- final overlay forecast

## Routing Logic

The routing layer works as follows:

1. each active model is scored online using recent excess loss relative to the best active model
2. scores are weighted by both time decay and regime similarity
3. a local-shrunk threshold `tau_t` is calibrated from recent routing scores
4. a routing set is formed by retaining models that remain competitive under the current state
5. calm and stress specialist branches are built from pre-specified pools
6. a stress probability is computed from the standardized market-state vector
7. calm and stress forecasts are combined
8. a second gate blends low-state and high-state branches
9. a conditional HAR floor is applied in stressed or high-dispersion conditions

### Specialist pools
- calm pool: `gru`, `har`, `xgb`
- stress pool: `garch_t`, `figarch`, `har`

## Evaluation Metrics

The implementation reports:

### Forecasting metrics
- QLIKE on the variance scale
- underprediction loss
- tail underprediction loss
- tail QLIKE
- RMSE on the volatility scale
- MAE on the volatility scale

### Routing diagnostics
- calm-branch usage rate
- stress-branch usage rate
- selected regret
- miss-best rate

### Statistical comparison
- asset-level Diebold–Mariano tests
- Newey–West HAC variance estimate for loss differentials

## Data Layout

The repository expects the two input files under `data/`:

- `data/multiasset_daily_10y_panel_model.csv`
- `data/master_daily_features_macro_dailyonly_raw_only.csv`

## Environment Setup

Create a clean Python environment and install the core dependencies.

### Option 1: virtual environment

```bash
python -m venv .venv
