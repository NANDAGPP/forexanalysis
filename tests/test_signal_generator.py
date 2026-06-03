"""Tests for the Signal Generator."""

import pytest
from src.models.signal_generator import SignalGenerator


@pytest.fixture
def generator():
    return SignalGenerator()


@pytest.fixture
def mock_ml_forecast():
    return {
        "current_price": 1.1050,
        "forecast_price": 1.1080,
        "price_change": 0.003,
        "price_change_pct": 0.27,
        "direction": "UP",
        "confidence_score": 75.0,
    }


@pytest.fixture
def mock_tech_signal():
    return {
        "signal": "BUY",
        "strength": 25,
        "details": {
            "rsi": {"value": 45, "signal": "Slightly bullish"},
            "macd": {"signal": "Bullish crossover"},
            "ema": {"signal": "Mild uptrend"},
        },
    }


@pytest.fixture
def mock_sr_levels():
    return {
        "current_price": 1.1050,
        "support": [1.1020, 1.1000, 1.0980],
        "resistance": [1.1080, 1.1100, 1.1120],
    }


def test_generate_signal(generator, mock_ml_forecast, mock_tech_signal, mock_sr_levels):
    """Test signal generation."""
    signal = generator.generate_signal("EUR/USD", mock_ml_forecast, mock_tech_signal, mock_sr_levels)
    assert signal["pair"] == "EUR/USD"
    assert "action" in signal
    assert "confidence" in signal
    assert "combined_score" in signal
    assert "risk_management" in signal
    assert "market_context" in signal


def test_signal_with_bearish_input(generator, mock_sr_levels):
    """Test signal with bearish ML + TA."""
    bearish_ml = {
        "current_price": 1.1050,
        "forecast_price": 1.1000,
        "price_change": -0.005,
        "price_change_pct": -0.45,
        "direction": "DOWN",
        "confidence_score": 80.0,
    }
    bearish_ta = {"signal": "SELL", "strength": -30, "details": {}}

    signal = generator.generate_signal("EUR/USD", bearish_ml, bearish_ta, mock_sr_levels)
    assert "SELL" in signal["action"]
    assert signal["combined_score"] < 0


def test_risk_management(generator, mock_ml_forecast, mock_tech_signal, mock_sr_levels):
    """Test risk parameters are calculated correctly."""
    signal = generator.generate_signal("EUR/USD", mock_ml_forecast, mock_tech_signal, mock_sr_levels)
    risk = signal["risk_management"]
    if risk.get("entry"):
        assert risk["stop_loss"] is not None
        assert risk["take_profit"] is not None
        assert risk["risk_reward"] >= 0


def test_portfolio_signals(generator, mock_ml_forecast, mock_tech_signal, mock_sr_levels):
    """Test portfolio-level signal analysis."""
    signals = []
    for pair in ["EUR/USD", "GBP/USD", "USD/JPY"]:
        s = generator.generate_signal(pair, mock_ml_forecast, mock_tech_signal, mock_sr_levels)
        signals.append(s)

    portfolio = generator.get_portfolio_signals(signals)
    assert portfolio["total_pairs_analyzed"] == 3
    assert "market_bias" in portfolio
    assert "top_opportunities" in portfolio
