# AGENTS.md

## Cursor Cloud specific instructions

This repository is a **Forex ML Trading Agent** — a Python-based application that combines machine learning forecasting with real-time technical analysis to generate forex trading signals.

### Architecture

- `src/data/fetcher.py` — Real-time + historical data from Yahoo Finance (yfinance)
- `src/models/forecaster.py` — Ensemble ML (Random Forest + Gradient Boosting + LSTM) for price prediction
- `src/models/signal_generator.py` — Combines ML output + technical indicators into trade signals
- `src/analysis/technical.py` — Technical indicators engine (RSI, MACD, Bollinger, EMA, ADX, etc.)
- `src/api/app.py` — Flask + WebSocket server with REST API
- `src/templates/dashboard.html` — Real-time web dashboard

### Running the application

```bash
python3 run.py
```
Dashboard at http://localhost:5000. API at http://localhost:5000/api/pairs.

### Running tests

```bash
python3 -m pytest tests/ -v
```

### Key API Endpoints

- `GET /api/pairs` — List available forex pairs
- `GET /api/analysis/<pair>` — Full analysis (pair format: EUR-USD)
- `GET /api/forecast/<pair>` — ML forecast only
- `GET /api/signals?pairs=EUR/USD,GBP/USD` — Multi-pair trading signals
- `GET /api/historical/<pair>?period=6mo&interval=1d` — Historical OHLCV data

### Notes

- First call to `/api/analysis` or `/api/forecast` for a pair triggers model training (~10-20s). Subsequent calls use cached models.
- Models are saved to `saved_models/` directory and auto-loaded on restart.
- yfinance data may be delayed or unavailable on weekends/holidays.
- LSTM training requires TensorFlow; if unavailable the system falls back to the RF+GB ensemble.
