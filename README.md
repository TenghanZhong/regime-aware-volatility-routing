# Regime-Aware Volatility Routing

This repository contains code for a regime-aware specialist routing framework for next-day ETF volatility forecasting.

The implementation supports a rolling walk-forward backtest with a mixed model pool, risk-sensitive online model scoring, regime-aware routing, state-dependent forecast combination, routing diagnostics, and Diebold–Mariano tests.

## Overview

The main script implements the following components:

- next-day realized variance / volatility forecasting
- candidate model pool:
  - HAR-RV
  - GARCH-t
  - FIGARCH
  - GRU
  - XGBoost (with sklearn fallback)
  - optional EGARCH
- risk-sensitive best-model loss:
  - `QLIKE + lambda_under * underprediction_penalty`
- rolling-best and static-best baselines
- naive VIX-switch baseline:
  - VIX > threshold -> GARCH-t
  - otherwise -> GRU
- regime-aware online model scoring with local-shrunk tau calibration
- two-sided state-gated blend policy
- conditional HAR floor in stressed / high-dispersion states
- routing diagnostics
- cross-asset Diebold–Mariano tests with Newey–West HAC variance

## Main script

Example filename:

```bash
regime_aware_backtest.py
