"""
Real-time and historical forex data fetcher.
Uses yfinance for market data and provides real-time price updates.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import logging

logger = logging.getLogger(__name__)

FOREX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
}


class ForexDataFetcher:
    """Fetches forex data from Yahoo Finance (real-time and historical)."""

    def __init__(self):
        self._cache = {}
        self._live_prices = {}
        self._running = False
        self._thread = None

    def get_historical_data(
        self, pair: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a forex pair.

        Args:
            pair: Forex pair name (e.g., "EUR/USD")
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk)
        """
        ticker_symbol = FOREX_PAIRS.get(pair)
        if not ticker_symbol:
            raise ValueError(f"Unknown pair: {pair}. Available: {list(FOREX_PAIRS.keys())}")

        cache_key = f"{pair}_{period}_{interval}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < 300:
                return cached_data

        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {pair}")
                return pd.DataFrame()

            df.index = pd.to_datetime(df.index)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.dropna(inplace=True)

            self._cache[cache_key] = (datetime.now(), df)
            logger.info(f"Fetched {len(df)} bars for {pair} ({period}, {interval})")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            return pd.DataFrame()

    def get_realtime_price(self, pair: str) -> dict:
        """Get the latest real-time price for a forex pair."""
        ticker_symbol = FOREX_PAIRS.get(pair)
        if not ticker_symbol:
            raise ValueError(f"Unknown pair: {pair}")

        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.fast_info
            history = ticker.history(period="1d", interval="1m")

            if history.empty:
                history = ticker.history(period="5d", interval="1h")

            if not history.empty:
                latest = history.iloc[-1]
                prev_close = history.iloc[-2]["Close"] if len(history) > 1 else latest["Close"]
                change = latest["Close"] - prev_close
                change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

                return {
                    "pair": pair,
                    "price": float(latest["Close"]),
                    "open": float(latest["Open"]),
                    "high": float(latest["High"]),
                    "low": float(latest["Low"]),
                    "change": float(change),
                    "change_pct": float(change_pct),
                    "timestamp": str(history.index[-1]),
                    "bid": float(latest["Close"] - 0.0001),
                    "ask": float(latest["Close"] + 0.0001),
                }
            return {"pair": pair, "price": None, "error": "No data available"}

        except Exception as e:
            logger.error(f"Error getting real-time price for {pair}: {e}")
            return {"pair": pair, "price": None, "error": str(e)}

    def get_multiple_pairs_data(
        self, pairs: list = None, period: str = "6mo", interval: str = "1d"
    ) -> dict:
        """Fetch historical data for multiple pairs."""
        if pairs is None:
            pairs = list(FOREX_PAIRS.keys())

        results = {}
        for pair in pairs:
            results[pair] = self.get_historical_data(pair, period, interval)
        return results

    def start_live_feed(self, pairs: list = None, callback=None, update_interval: int = 30):
        """Start a background thread that fetches live prices periodically."""
        if pairs is None:
            pairs = list(FOREX_PAIRS.keys())[:5]

        self._running = True

        def _feed_loop():
            while self._running:
                for pair in pairs:
                    price_data = self.get_realtime_price(pair)
                    self._live_prices[pair] = price_data
                    if callback:
                        callback(price_data)
                time.sleep(update_interval)

        self._thread = threading.Thread(target=_feed_loop, daemon=True)
        self._thread.start()
        logger.info(f"Live feed started for {pairs}")

    def stop_live_feed(self):
        """Stop the background live feed."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Live feed stopped")

    @property
    def live_prices(self) -> dict:
        return self._live_prices.copy()

    @staticmethod
    def get_available_pairs() -> dict:
        return FOREX_PAIRS.copy()
