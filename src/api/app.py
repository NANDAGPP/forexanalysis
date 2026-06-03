"""
Flask Web Application for Forex ML Trading Agent.
Provides REST API and real-time WebSocket dashboard.
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.fetcher import ForexDataFetcher, FOREX_PAIRS
from src.analysis.technical import TechnicalAnalyzer
from src.models.forecaster import ForexForecaster
from src.models.signal_generator import SignalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "forex-ml-agent-secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

fetcher = ForexDataFetcher()
analyzer = TechnicalAnalyzer()
signal_gen = SignalGenerator()
forecasters = {}

analysis_cache = {}
CACHE_TTL = 60


def get_or_train_forecaster(pair: str) -> ForexForecaster:
    """Get a trained forecaster for a pair, training if necessary."""
    if pair in forecasters and forecasters[pair].is_trained:
        return forecasters[pair]

    fc = ForexForecaster(lookback=60, forecast_horizon=5)

    if fc.load_model(pair):
        forecasters[pair] = fc
        return fc

    df = fetcher.get_historical_data(pair, period="1y", interval="1d")
    if df.empty or len(df) < 150:
        logger.warning(f"Insufficient data to train model for {pair}")
        return None

    try:
        fc.train(df, pair)
        fc.save_model(pair)
        forecasters[pair] = fc
        return fc
    except Exception as e:
        logger.error(f"Failed to train model for {pair}: {e}")
        return None


@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("dashboard.html", pairs=list(FOREX_PAIRS.keys()))


@app.route("/api/pairs")
def get_pairs():
    """Get available forex pairs."""
    return jsonify({"pairs": list(FOREX_PAIRS.keys())})


@app.route("/api/price/<path:pair>")
def get_price(pair):
    """Get real-time price for a pair."""
    pair = pair.replace("-", "/")
    price_data = fetcher.get_realtime_price(pair)
    return jsonify(price_data)


@app.route("/api/analysis/<path:pair>")
def get_analysis(pair):
    """Get full analysis for a pair including technicals, forecast, and signals."""
    pair = pair.replace("-", "/")

    cache_key = f"analysis_{pair}"
    if cache_key in analysis_cache:
        cached_time, cached_data = analysis_cache[cache_key]
        if (datetime.now() - cached_time).seconds < CACHE_TTL:
            return jsonify(cached_data)

    try:
        df = fetcher.get_historical_data(pair, period="6mo", interval="1d")
        if df.empty:
            return jsonify({"error": f"No data available for {pair}"}), 404

        indicators_df = analyzer.compute_all_indicators(df)
        tech_signal = analyzer.get_signal_strength(indicators_df)
        sr_levels = analyzer.get_support_resistance(indicators_df)
        market_summary = analyzer.get_market_summary(df, pair)

        fc = get_or_train_forecaster(pair)
        ml_forecast = None
        if fc:
            try:
                ml_forecast = fc.predict(df)
            except Exception as e:
                logger.error(f"Forecast error for {pair}: {e}")

        trading_signal = None
        if ml_forecast:
            trading_signal = signal_gen.generate_signal(
                pair, ml_forecast, tech_signal, sr_levels
            )

        result = {
            "pair": pair,
            "timestamp": datetime.now().isoformat(),
            "price": {
                "current": float(df["close"].iloc[-1]),
                "open": float(df["open"].iloc[-1]),
                "high": float(df["high"].iloc[-1]),
                "low": float(df["low"].iloc[-1]),
            },
            "technical_analysis": tech_signal,
            "support_resistance": sr_levels,
            "ml_forecast": ml_forecast,
            "trading_signal": trading_signal,
            "market_summary": market_summary,
        }

        analysis_cache[cache_key] = (datetime.now(), result)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Analysis error for {pair}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast/<path:pair>")
def get_forecast(pair):
    """Get ML forecast for a pair."""
    pair = pair.replace("-", "/")

    try:
        df = fetcher.get_historical_data(pair, period="1y", interval="1d")
        if df.empty:
            return jsonify({"error": f"No data available for {pair}"}), 404

        fc = get_or_train_forecaster(pair)
        if not fc:
            return jsonify({"error": "Could not train model - insufficient data"}), 400

        forecast = fc.predict(df)
        return jsonify(forecast)

    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals")
def get_all_signals():
    """Get trading signals for all major pairs."""
    pairs = request.args.get("pairs", "").split(",") if request.args.get("pairs") else list(FOREX_PAIRS.keys())[:5]

    signals = []
    for pair in pairs:
        pair = pair.strip()
        if pair not in FOREX_PAIRS:
            continue

        try:
            df = fetcher.get_historical_data(pair, period="6mo", interval="1d")
            if df.empty:
                continue

            indicators_df = analyzer.compute_all_indicators(df)
            tech_signal = analyzer.get_signal_strength(indicators_df)
            sr_levels = analyzer.get_support_resistance(indicators_df)

            fc = get_or_train_forecaster(pair)
            if fc:
                ml_forecast = fc.predict(df)
                signal = signal_gen.generate_signal(pair, ml_forecast, tech_signal, sr_levels)
                signals.append(signal)
        except Exception as e:
            logger.error(f"Signal generation error for {pair}: {e}")

    portfolio = signal_gen.get_portfolio_signals(signals)
    return jsonify({"signals": signals, "portfolio_summary": portfolio})


@app.route("/api/historical/<path:pair>")
def get_historical(pair):
    """Get historical OHLCV data."""
    pair = pair.replace("-", "/")
    period = request.args.get("period", "6mo")
    interval = request.args.get("interval", "1d")

    df = fetcher.get_historical_data(pair, period=period, interval=interval)
    if df.empty:
        return jsonify({"error": "No data available"}), 404

    data = df.reset_index().to_dict(orient="records")
    for record in data:
        if "Date" in record:
            record["Date"] = str(record["Date"])
        elif "Datetime" in record:
            record["Datetime"] = str(record["Datetime"])
    return jsonify({"pair": pair, "period": period, "interval": interval, "data": data})


@socketio.on("connect")
def handle_connect():
    """Handle WebSocket connection."""
    logger.info("Client connected")
    emit("connected", {"status": "connected", "pairs": list(FOREX_PAIRS.keys())})


@socketio.on("subscribe")
def handle_subscribe(data):
    """Subscribe to real-time updates for a pair."""
    pair = data.get("pair", "EUR/USD")
    logger.info(f"Client subscribed to {pair}")

    def send_updates():
        while True:
            try:
                price_data = fetcher.get_realtime_price(pair)
                socketio.emit("price_update", price_data)
                time.sleep(30)
            except Exception as e:
                logger.error(f"WebSocket update error: {e}")
                break

    thread = threading.Thread(target=send_updates, daemon=True)
    thread.start()


@socketio.on("request_analysis")
def handle_analysis_request(data):
    """Handle real-time analysis request via WebSocket."""
    pair = data.get("pair", "EUR/USD")
    try:
        df = fetcher.get_historical_data(pair, period="6mo", interval="1d")
        if not df.empty:
            summary = analyzer.get_market_summary(df, pair)
            emit("analysis_update", summary)
    except Exception as e:
        emit("analysis_error", {"error": str(e)})


def create_app():
    """Application factory."""
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Forex ML Trading Agent on port {port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=True, allow_unsafe_werkzeug=True)
