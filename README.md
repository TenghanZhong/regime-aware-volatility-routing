# Regime-Aware Volatility Routing

This repository contains code for a regime-aware specialist routing framework for next-day ETF volatility forecasting.

The implementation is designed as a rolling walk-forward backtest with a mixed model pool, risk-sensitive online model scoring, regime-aware routing, state-dependent forecast combination, routing diagnostics, and Diebold–Mariano tests.

## Main idea

The main script implements a pragmatic, paper-oriented backtest for:

1. next-day realized variance / volatility forecasting
2. candidate model pool:
   - HAR-RV
   - GARCH-t
   - FIGARCH
   - GRU
   - XGBoost (with sklearn fallback)
   - optional EGARCH, dropped by default
3. risk-sensitive best-model loss:
   - `QLIKE + lambda_under * UnderPenalty`
4. rolling-best and static-best baselines
5. naive VIX-switch baseline:
   - if `VIX > threshold`, use GARCH-t
   - otherwise, use GRU
6. regime-aware online model scoring with local-shrunk tau calibration
7. two-sided state-gated blend policy:
   - low-state branch blends `rolling_best` and calm specialist
   - high-state branch blends `combo` and stress specialist
   - `omega_t` blends low/high branches
   - HAR acts as a conditional floor in stressed / high-dispersion states
8. regime-conditional aggregation:
   - calm specialist pool: GRU / HAR / XGBoost
   - stress specialist pool: GARCH-t / FIGARCH / HAR
9. routing diagnostics:
   - compact 4-metric regime table
   - detailed specialist day-usage table
10. cross-asset Diebold–Mariano tests with Newey–West HAC variance

## Method overview

The backtest constructs the next-day variance target and forecasting features internally.

### Target construction

From the panel file, the script builds the target as:

- `rv_var = gk_proxy^2`
- `rv_vol = sqrt(rv_var)`
- `y_next_var = rv_var shifted by -1`
- `y_next_vol = sqrt(y_next_var)`

QLIKE is evaluated on the variance scale, while RMSE and MAE are also reported on the corresponding volatility scale.

### Forecasting features

The implementation builds features such as:

- HAR-style variance inputs
- lagged variance terms
- return lags and absolute-return lags
- rolling variance means and standard deviations
- log-variance features
- volatility-of-volatility features
- rolling volatility and EWMA volatility
- macro-financial state features:
  - `log_vix`
  - `log_vix_vxv`
  - `d5_log_vix`
  - `term_spread`
  - `hy_oas`
  - `vix_raw`
- a 20-day rolling volatility summary `rv20d`

### Default configuration

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
- upper quantile for stress aggregation: `0.75`
- winsorization bounds: `0.10` and `0.90`
- HAR floor stress threshold: `0.65`
- HAR floor dispersion threshold: `0.20`
- random seed: `42`

### Supported model pool

The backtest supports the following models:

- `har`
- `garch_t`
- `figarch`
- `gru`
- `xgb`

Optional:

- `egarch`

The internal model-pool logic allows `egarch` to be either dropped or activated depending on configuration.

### Methods compared

The backtest includes the following benchmark and policy outputs.

**Single-model comparators**
- HAR-RV
- GARCH-t
- FIGARCH
- GRU
- XGBoost

**Adaptive baselines**
- static-best
- rolling-best
- naive VIX-switch

**Routing outputs**
- calm branch
- stress branch
- regime-conditional combo branch
- low-state branch
- high-state branch
- final overlay / routed forecast

### Routing logic

The routing layer works as follows:

1. Each active model is scored online using recent excess loss relative to the best active model.
2. The score uses time decay and regime similarity.
3. A local-shrunk threshold `tau_t` is calibrated using recent routing scores.
4. A routing set is selected by retaining models whose scores remain close to recent competitive levels.
5. Calm and stress specialist branches are built from model pools:
   - calm: `gru`, `har`, `xgb`
   - stress: `garch_t`, `figarch`, `har`
6. A stress probability is computed from the standardized market-state vector.
7. Calm and stress branch forecasts are combined.
8. A second gate blends the low-state and high-state branches.
9. A conditional HAR floor is applied in stressed or high-dispersion conditions.

### Evaluation metrics

The script reports:

- QLIKE on the variance scale
- underprediction loss
- tail underprediction loss
- tail QLIKE

For routing behavior, the script reports:

- calm-branch usage rate
- stress-branch usage rate
- selected regret
- miss-best rate

It also computes asset-level Diebold–Mariano tests using a Newey–West HAC variance estimate.




