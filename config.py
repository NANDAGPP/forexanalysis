"""Application configuration."""

import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "forex-ml-agent-dev-key")
    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"
    PORT = int(os.environ.get("PORT", 5000))
    HOST = os.environ.get("HOST", "0.0.0.0")

    # Data settings
    DEFAULT_PERIOD = "6mo"
    DEFAULT_INTERVAL = "1d"
    CACHE_TTL_SECONDS = 60

    # ML settings
    LOOKBACK_PERIOD = 60
    FORECAST_HORIZON = 5
    RETRAIN_INTERVAL_HOURS = 24

    # Risk management
    RISK_PER_TRADE = 0.02
    MAX_POSITIONS = 3
    DEFAULT_ACCOUNT_BALANCE = 10000

    # Real-time update interval (seconds)
    LIVE_UPDATE_INTERVAL = 30
