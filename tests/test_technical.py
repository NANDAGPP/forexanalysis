"""Tests for the Technical Analysis engine."""

import pytest
import pandas as pd
import numpy as np
from src.analysis.technical import TechnicalAnalyzer


@pytest.fixture
def analyzer():
    return TechnicalAnalyzer()


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.002)
    high = close + np.abs(np.random.randn(n) * 0.001)
    low = close - np.abs(np.random.randn(n) * 0.001)
    open_price = close + np.random.randn(n) * 0.0005
    volume = np.random.randint(1000, 100000, n)

    df = pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df


def test_compute_indicators(analyzer, sample_ohlcv):
    """Test that all indicators are computed without errors."""
    result = analyzer.compute_all_indicators(sample_ohlcv)
    assert not result.empty
    assert "rsi" in result.columns
    assert "ema_9" in result.columns
    assert "ema_21" in result.columns
    assert "ema_50" in result.columns
    assert "atr" in result.columns


def test_signal_strength(analyzer, sample_ohlcv):
    """Test signal strength computation."""
    indicators_df = analyzer.compute_all_indicators(sample_ohlcv)
    signal = analyzer.get_signal_strength(indicators_df)
    assert "signal" in signal
    assert "strength" in signal
    assert signal["signal"] in ["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"]
    assert -100 <= signal["strength"] <= 100


def test_support_resistance(analyzer, sample_ohlcv):
    """Test support and resistance level identification."""
    sr = analyzer.get_support_resistance(sample_ohlcv)
    assert "support" in sr
    assert "resistance" in sr
    assert "current_price" in sr
    assert isinstance(sr["support"], list)
    assert isinstance(sr["resistance"], list)


def test_market_summary(analyzer, sample_ohlcv):
    """Test market summary generation."""
    summary = analyzer.get_market_summary(sample_ohlcv, "EUR/USD")
    assert summary["pair"] == "EUR/USD"
    assert "current_price" in summary
    assert "signal" in summary
    assert "recommendation" in summary
    assert summary["recommendation"]["action"] in ["BUY", "SELL", "HOLD"]


def test_insufficient_data(analyzer):
    """Test behavior with insufficient data."""
    short_df = pd.DataFrame(
        {"open": [1.1], "high": [1.11], "low": [1.09], "close": [1.1], "volume": [1000]},
        index=pd.date_range("2024-01-01", periods=1),
    )
    result = analyzer.compute_all_indicators(short_df)
    assert len(result) <= 1

    signal = analyzer.get_signal_strength(short_df)
    assert signal["signal"] == "NEUTRAL"
