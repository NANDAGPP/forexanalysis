"""Tests for the ML Forecaster."""

import pytest
import pandas as pd
import numpy as np
from src.models.forecaster import ForexForecaster


@pytest.fixture
def forecaster():
    return ForexForecaster(lookback=30, forecast_horizon=5)


@pytest.fixture
def sample_data():
    """Generate sufficient synthetic data for model training."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
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


def test_prepare_features(forecaster, sample_data):
    """Test feature engineering pipeline."""
    features = forecaster.prepare_features(sample_data)
    assert not features.empty
    assert "returns" in features.columns
    assert "rsi" in features.columns
    assert "momentum_5" in features.columns
    assert "bb_position" in features.columns
    assert len(features) < len(sample_data)


def test_train_models(forecaster, sample_data):
    """Test model training pipeline."""
    metrics = forecaster.train(sample_data, pair="TEST/USD")
    assert forecaster.is_trained
    assert "random_forest" in metrics
    assert "gradient_boosting" in metrics
    assert metrics["random_forest"]["mae"] > 0
    assert metrics["gradient_boosting"]["mae"] > 0


def test_predict(forecaster, sample_data):
    """Test prediction after training."""
    forecaster.train(sample_data, pair="TEST/USD")
    prediction = forecaster.predict(sample_data)
    assert "current_price" in prediction
    assert "forecast_price" in prediction
    assert "direction" in prediction
    assert "confidence_score" in prediction
    assert prediction["direction"] in ["UP", "DOWN", "FLAT"]
    assert 0 <= prediction["confidence_score"] <= 100
    assert "model_predictions" in prediction


def test_predict_without_training(forecaster, sample_data):
    """Test that predict raises error without training."""
    with pytest.raises(RuntimeError):
        forecaster.predict(sample_data)


def test_insufficient_data(forecaster):
    """Test behavior with insufficient data."""
    short_df = pd.DataFrame(
        {"open": [1.1] * 10, "high": [1.11] * 10, "low": [1.09] * 10,
         "close": [1.1] * 10, "volume": [1000] * 10},
        index=pd.date_range("2024-01-01", periods=10),
    )
    with pytest.raises(ValueError):
        forecaster.train(short_df)
