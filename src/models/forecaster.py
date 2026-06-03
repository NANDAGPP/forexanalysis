"""
ML-based Forex Price Forecaster.
Implements LSTM neural network and ensemble methods for price prediction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)


class ForexForecaster:
    """
    Ensemble ML forecaster combining LSTM and tree-based models
    for high-accuracy forex price prediction.
    """

    def __init__(self, lookback: int = 60, forecast_horizon: int = 5):
        """
        Args:
            lookback: Number of past time steps to use as features
            forecast_horizon: Number of future steps to predict
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_model = None
        self.rf_model = None
        self.gb_model = None
        self.is_trained = False
        self._training_metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature-engineered dataframe from OHLCV data."""
        features = df.copy()

        features["returns"] = features["close"].pct_change()
        features["log_returns"] = np.log(features["close"] / features["close"].shift(1))

        for window in [5, 10, 20, 50]:
            features[f"sma_{window}"] = features["close"].rolling(window).mean()
            features[f"std_{window}"] = features["close"].rolling(window).std()
            features[f"returns_mean_{window}"] = features["returns"].rolling(window).mean()

        features["momentum_5"] = features["close"] / features["close"].shift(5) - 1
        features["momentum_10"] = features["close"] / features["close"].shift(10) - 1
        features["momentum_20"] = features["close"] / features["close"].shift(20) - 1

        features["rsi"] = self._compute_rsi(features["close"], 14)

        features["bb_upper"] = features["sma_20"] + 2 * features["std_20"]
        features["bb_lower"] = features["sma_20"] - 2 * features["std_20"]
        features["bb_position"] = (features["close"] - features["bb_lower"]) / (
            features["bb_upper"] - features["bb_lower"]
        )

        features["atr"] = self._compute_atr(features, 14)

        features["hl_range"] = (features["high"] - features["low"]) / features["close"]
        features["oc_range"] = (features["close"] - features["open"]) / features["close"]

        features["day_of_week"] = features.index.dayofweek if hasattr(features.index, "dayofweek") else 0
        features["hour"] = features.index.hour if hasattr(features.index, "hour") else 0

        features.dropna(inplace=True)
        return features

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _create_sequences(self, data: np.ndarray, target: np.ndarray):
        """Create input sequences for LSTM model."""
        X, y = [], []
        for i in range(self.lookback, len(data) - self.forecast_horizon + 1):
            X.append(data[i - self.lookback:i])
            y.append(target[i:i + self.forecast_horizon])
        return np.array(X), np.array(y)

    def _create_flat_features(self, data: np.ndarray, target: np.ndarray):
        """Create flattened feature matrix for tree-based models."""
        X, y = [], []
        for i in range(self.lookback, len(data) - self.forecast_horizon + 1):
            X.append(data[i - self.lookback:i].flatten())
            y.append(target[i:i + self.forecast_horizon].mean())
        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame, pair: str = "unknown") -> dict:
        """
        Train all models on historical data.

        Returns training metrics dict.
        """
        logger.info(f"Training forecaster for {pair} with {len(df)} samples")

        features_df = self.prepare_features(df)
        if len(features_df) < self.lookback + self.forecast_horizon + 50:
            raise ValueError(f"Insufficient data: need at least {self.lookback + self.forecast_horizon + 50} rows")

        feature_cols = [c for c in features_df.columns if c not in ["open", "high", "low", "close", "volume"]]
        target = features_df["close"].values

        feature_data = features_df[feature_cols].values
        self.feature_scaler.fit(feature_data)
        scaled_features = self.feature_scaler.transform(feature_data)

        self.scaler.fit(target.reshape(-1, 1))
        scaled_target = self.scaler.transform(target.reshape(-1, 1)).flatten()

        split_idx = int(len(scaled_features) * 0.8)

        X_flat, y_flat = self._create_flat_features(scaled_features, scaled_target)
        train_size = int(len(X_flat) * 0.8)

        X_train_flat, X_test_flat = X_flat[:train_size], X_flat[train_size:]
        y_train_flat, y_test_flat = y_flat[:train_size], y_flat[train_size:]

        logger.info("Training Random Forest model...")
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model.fit(X_train_flat, y_train_flat)
        rf_pred = self.rf_model.predict(X_test_flat)
        rf_mae = mean_absolute_error(y_test_flat, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test_flat, rf_pred))

        logger.info("Training Gradient Boosting model...")
        self.gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.gb_model.fit(X_train_flat, y_train_flat)
        gb_pred = self.gb_model.predict(X_test_flat)
        gb_mae = mean_absolute_error(y_test_flat, gb_pred)
        gb_rmse = np.sqrt(mean_squared_error(y_test_flat, gb_pred))

        try:
            self._train_lstm(scaled_features, scaled_target)
            lstm_trained = True
        except Exception as e:
            logger.warning(f"LSTM training failed, using ensemble only: {e}")
            lstm_trained = False

        self.is_trained = True
        self._feature_cols = feature_cols

        self._training_metrics = {
            "pair": pair,
            "samples": len(features_df),
            "train_size": train_size,
            "test_size": len(X_test_flat),
            "random_forest": {"mae": float(rf_mae), "rmse": float(rf_rmse)},
            "gradient_boosting": {"mae": float(gb_mae), "rmse": float(gb_rmse)},
            "lstm_trained": lstm_trained,
            "lookback": self.lookback,
            "forecast_horizon": self.forecast_horizon,
        }

        logger.info(f"Training complete. RF MAE: {rf_mae:.6f}, GB MAE: {gb_mae:.6f}")
        return self._training_metrics

    def _train_lstm(self, features: np.ndarray, target: np.ndarray):
        """Train LSTM model for sequence prediction."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            X_seq, y_seq = self._create_sequences(features, target)
            train_size = int(len(X_seq) * 0.8)

            X_train, X_test = X_seq[:train_size], X_seq[train_size:]
            y_train, y_test = y_seq[:train_size], y_seq[train_size:]

            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
                Dropout(0.2),
                Bidirectional(LSTM(32, return_sequences=False)),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(self.forecast_horizon),
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5),
            ]

            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=0,
            )

            self.lstm_model = model
            logger.info("LSTM model trained successfully")

        except ImportError:
            logger.warning("TensorFlow not available, skipping LSTM")
            raise

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Generate price forecast for next N periods.

        Returns prediction dict with forecasted prices and confidence intervals.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        features_df = self.prepare_features(df)
        feature_cols = self._feature_cols
        feature_data = features_df[feature_cols].values

        scaled_features = self.feature_scaler.transform(feature_data)
        current_price = df["close"].iloc[-1]

        latest_flat = scaled_features[-self.lookback:].flatten().reshape(1, -1)

        rf_pred_scaled = self.rf_model.predict(latest_flat)[0]
        gb_pred_scaled = self.gb_model.predict(latest_flat)[0]

        lstm_pred_scaled = None
        if self.lstm_model is not None:
            latest_seq = scaled_features[-self.lookback:].reshape(1, self.lookback, -1)
            lstm_pred_scaled = self.lstm_model.predict(latest_seq, verbose=0)[0]

        if lstm_pred_scaled is not None:
            ensemble_scaled = (
                0.3 * rf_pred_scaled +
                0.3 * gb_pred_scaled +
                0.4 * np.mean(lstm_pred_scaled)
            )
            lstm_prices = self.scaler.inverse_transform(
                lstm_pred_scaled.reshape(-1, 1)
            ).flatten()
        else:
            ensemble_scaled = 0.5 * rf_pred_scaled + 0.5 * gb_pred_scaled
            lstm_prices = None

        ensemble_price = self.scaler.inverse_transform([[ensemble_scaled]])[0][0]

        price_change = ensemble_price - current_price
        price_change_pct = (price_change / current_price) * 100
        direction = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT"

        rf_price = self.scaler.inverse_transform([[rf_pred_scaled]])[0][0]
        gb_price = self.scaler.inverse_transform([[gb_pred_scaled]])[0][0]

        predictions = [rf_price, gb_price]
        if lstm_prices is not None:
            predictions.append(np.mean(lstm_prices))
        std_dev = np.std(predictions)

        confidence_upper = ensemble_price + 2 * std_dev
        confidence_lower = ensemble_price - 2 * std_dev

        model_agreement = 1 - (std_dev / current_price)
        confidence_score = max(0, min(100, model_agreement * 100))

        return {
            "current_price": float(current_price),
            "forecast_price": float(ensemble_price),
            "price_change": float(price_change),
            "price_change_pct": float(price_change_pct),
            "direction": direction,
            "confidence_score": float(confidence_score),
            "confidence_interval": {
                "upper": float(confidence_upper),
                "lower": float(confidence_lower),
            },
            "model_predictions": {
                "random_forest": float(rf_price),
                "gradient_boosting": float(gb_price),
                "lstm": float(np.mean(lstm_prices)) if lstm_prices is not None else None,
                "ensemble": float(ensemble_price),
            },
            "forecast_horizon": self.forecast_horizon,
            "training_metrics": self._training_metrics,
        }

    def save_model(self, pair: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("No trained model to save")

        pair_clean = pair.replace("/", "_")
        path = os.path.join(MODEL_DIR, pair_clean)
        os.makedirs(path, exist_ok=True)

        joblib.dump(self.rf_model, os.path.join(path, "rf_model.pkl"))
        joblib.dump(self.gb_model, os.path.join(path, "gb_model.pkl"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        joblib.dump(self.feature_scaler, os.path.join(path, "feature_scaler.pkl"))
        joblib.dump(self._feature_cols, os.path.join(path, "feature_cols.pkl"))

        if self.lstm_model is not None:
            self.lstm_model.save(os.path.join(path, "lstm_model.keras"))

        logger.info(f"Models saved to {path}")

    def load_model(self, pair: str) -> bool:
        """Load previously saved model from disk."""
        pair_clean = pair.replace("/", "_")
        path = os.path.join(MODEL_DIR, pair_clean)

        if not os.path.exists(path):
            return False

        try:
            self.rf_model = joblib.load(os.path.join(path, "rf_model.pkl"))
            self.gb_model = joblib.load(os.path.join(path, "gb_model.pkl"))
            self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
            self.feature_scaler = joblib.load(os.path.join(path, "feature_scaler.pkl"))
            self._feature_cols = joblib.load(os.path.join(path, "feature_cols.pkl"))

            lstm_path = os.path.join(path, "lstm_model.keras")
            if os.path.exists(lstm_path):
                import tensorflow as tf
                self.lstm_model = tf.keras.models.load_model(lstm_path)

            self.is_trained = True
            logger.info(f"Models loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
