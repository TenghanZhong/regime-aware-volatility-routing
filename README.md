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

## Target

QLIKE is naturally defined on variance-like targets. The script therefore uses next-day realized variance as the primary target, operationalized as:

```text
y_{t+1} = (gk_proxy_{t+1})^2
