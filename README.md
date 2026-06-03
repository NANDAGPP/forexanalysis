# Forex ML Trading Agent

A machine learning-powered forex trading agent with real-time market analysis, price forecasting, and automated trading signal generation.

## Features

- **ML Price Forecasting**: Ensemble model combining LSTM, Random Forest, and Gradient Boosting for high-accuracy 5-day price predictions
- **Real-Time Technical Analysis**: 15+ indicators including RSI, MACD, Bollinger Bands, EMA, ADX, Stochastic, Ichimoku
- **Trading Signal Generation**: Combined ML + TA signals with confidence scoring and risk management
- **Live Dashboard**: Real-time web interface with interactive candlestick charts, indicators, and recommendations
- **Multi-Pair Support**: 10 major forex pairs with portfolio-level analysis
- **Risk Management**: Automated stop-loss, take-profit, and position sizing calculations

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python3 run.py

# Run tests
python3 -m pytest tests/ -v
```

Open http://localhost:5000 for the trading dashboard.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/pairs` | List available forex pairs |
| `GET /api/analysis/EUR-USD` | Full analysis with technicals + ML forecast |
| `GET /api/forecast/EUR-USD` | ML price forecast only |
| `GET /api/signals` | Multi-pair trading signals |
| `GET /api/historical/EUR-USD` | Historical OHLCV data |

## Architecture

```
src/
├── api/app.py              # Flask + WebSocket server
├── data/fetcher.py         # Real-time data from Yahoo Finance
├── models/
│   ├── forecaster.py       # ML ensemble (LSTM + RF + GB)
│   └── signal_generator.py # Trading signal engine
├── analysis/technical.py   # Technical indicators (15+)
├── templates/dashboard.html # Real-time web dashboard
└── static/                 # Static assets
```

## ML Models

The forecaster uses an ensemble approach:
- **LSTM** (Bidirectional): Captures sequential patterns in price data
- **Random Forest**: Robust non-linear feature relationships
- **Gradient Boosting**: High-precision residual learning

Models are automatically trained on first request and cached for subsequent calls.

## Technical Indicators

RSI, MACD, Bollinger Bands, EMA (9/21/50), SMA 200, ATR, Stochastic, ADX, OBV, Ichimoku Cloud, Pivot Points, Support/Resistance levels.

## Supported Pairs

EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD, EUR/GBP, EUR/JPY, GBP/JPY
