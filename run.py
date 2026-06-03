"""Entry point for the Forex ML Trading Agent application."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.api.app import app, socketio
from config import Config


def main():
    """Start the application server."""
    print("=" * 60)
    print("  Forex ML Trading Agent")
    print("  Real-time Analysis & ML Forecasting")
    print("=" * 60)
    print(f"\n  Dashboard: http://localhost:{Config.PORT}")
    print(f"  API Docs:  http://localhost:{Config.PORT}/api/pairs")
    print("=" * 60)

    socketio.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
