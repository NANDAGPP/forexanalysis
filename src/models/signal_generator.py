"""
Trading Signal Generator.
Combines ML predictions with technical analysis to produce actionable trade signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generates comprehensive trading signals by combining:
    - ML model predictions (direction + confidence)
    - Technical indicator analysis
    - Support/resistance levels
    - Risk management parameters
    """

    def __init__(self, risk_per_trade: float = 0.02, max_positions: int = 3):
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.active_signals = []

    def generate_signal(
        self,
        pair: str,
        ml_forecast: dict,
        technical_signal: dict,
        support_resistance: dict,
    ) -> dict:
        """
        Generate a comprehensive trading signal combining ML and technical analysis.

        Args:
            pair: Currency pair name
            ml_forecast: Output from ForexForecaster.predict()
            technical_signal: Output from TechnicalAnalyzer.get_signal_strength()
            support_resistance: Support/resistance levels
        """
        ml_direction = ml_forecast.get("direction", "FLAT")
        ml_confidence = ml_forecast.get("confidence_score", 0)
        ml_change_pct = ml_forecast.get("price_change_pct", 0)

        ta_signal = technical_signal.get("signal", "NEUTRAL")
        ta_strength = technical_signal.get("strength", 0)

        ml_score = self._direction_to_score(ml_direction, ml_confidence, ml_change_pct)
        ta_score = ta_strength

        combined_score = 0.6 * ml_score + 0.4 * ta_score

        if ml_score > 0 and ta_score > 0:
            combined_score *= 1.2
        elif ml_score < 0 and ta_score < 0:
            combined_score *= 1.2

        combined_score = max(-100, min(100, combined_score))

        action, confidence = self._score_to_action(combined_score)

        current_price = ml_forecast.get("current_price", 0)
        risk_params = self._calculate_risk_params(
            action, current_price, support_resistance, ml_forecast
        )

        signal = {
            "pair": pair,
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "confidence": confidence,
            "combined_score": round(combined_score, 2),
            "ml_component": {
                "direction": ml_direction,
                "confidence": round(ml_confidence, 2),
                "forecast_price": ml_forecast.get("forecast_price"),
                "price_change_pct": round(ml_change_pct, 4),
            },
            "technical_component": {
                "signal": ta_signal,
                "strength": round(ta_strength, 2),
                "indicators": technical_signal.get("details", {}),
            },
            "risk_management": risk_params,
            "market_context": self._assess_market_context(technical_signal),
        }

        self.active_signals.append(signal)
        if len(self.active_signals) > 100:
            self.active_signals = self.active_signals[-50:]

        return signal

    def _direction_to_score(self, direction: str, confidence: float, change_pct: float) -> float:
        """Convert ML prediction direction and confidence to a normalized score."""
        if direction == "UP":
            base = min(50, abs(change_pct) * 100)
        elif direction == "DOWN":
            base = -min(50, abs(change_pct) * 100)
        else:
            base = 0

        confidence_factor = confidence / 100
        return base * confidence_factor

    def _score_to_action(self, score: float) -> tuple:
        """Convert combined score to action and confidence level."""
        abs_score = abs(score)

        if abs_score < 15:
            return "HOLD", "LOW"
        elif abs_score < 35:
            action = "BUY" if score > 0 else "SELL"
            return action, "MEDIUM"
        else:
            action = "STRONG BUY" if score > 0 else "STRONG SELL"
            return action, "HIGH"

    def _calculate_risk_params(
        self, action: str, current_price: float, sr_levels: dict, forecast: dict
    ) -> dict:
        """Calculate stop-loss, take-profit, and position sizing."""
        if action in ("HOLD",) or current_price == 0:
            return {"entry": None, "stop_loss": None, "take_profit": None, "risk_reward": None}

        is_buy = "BUY" in action
        supports = sr_levels.get("support", [])
        resistances = sr_levels.get("resistance", [])

        if is_buy:
            stop_loss = supports[0] if supports else current_price * 0.995
            take_profit = resistances[0] if resistances else current_price * 1.015
        else:
            stop_loss = resistances[0] if resistances else current_price * 1.005
            take_profit = supports[0] if supports else current_price * 0.985

        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

        pip_value = 0.0001 if "JPY" not in str(sr_levels.get("pair", "")) else 0.01
        stop_pips = abs(current_price - stop_loss) / pip_value

        return {
            "entry": round(current_price, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "risk_reward": risk_reward,
            "stop_pips": round(stop_pips, 1),
            "recommended_lot_size": self._calculate_lot_size(stop_pips),
        }

    def _calculate_lot_size(self, stop_pips: float, account_balance: float = 10000) -> float:
        """Calculate recommended lot size based on risk management."""
        if stop_pips <= 0:
            return 0.01
        risk_amount = account_balance * self.risk_per_trade
        pip_value_per_lot = 10
        lot_size = risk_amount / (stop_pips * pip_value_per_lot)
        return round(max(0.01, min(lot_size, 1.0)), 2)

    def _assess_market_context(self, technical_signal: dict) -> dict:
        """Assess overall market context."""
        details = technical_signal.get("details", {})

        adx_info = details.get("adx", {})
        trend_strength = adx_info.get("trend_strength", "Unknown")

        volatility = "Normal"
        if "bollinger" in details:
            bb_pos = details["bollinger"].get("position", 0.5)
            if bb_pos > 0.9 or bb_pos < 0.1:
                volatility = "High"
            elif 0.3 < bb_pos < 0.7:
                volatility = "Low"

        if trend_strength == "Strong":
            regime = "Trending"
        elif trend_strength == "Weak":
            regime = "Ranging"
        else:
            regime = "Transitioning"

        return {
            "trend_strength": trend_strength,
            "volatility": volatility,
            "market_regime": regime,
        }

    def get_portfolio_signals(self, signals: list) -> dict:
        """Analyze multiple pair signals for portfolio-level decisions."""
        buy_signals = [s for s in signals if "BUY" in s.get("action", "")]
        sell_signals = [s for s in signals if "SELL" in s.get("action", "")]
        hold_signals = [s for s in signals if s.get("action") == "HOLD"]

        best_opportunities = sorted(
            [s for s in signals if s.get("action") != "HOLD"],
            key=lambda x: abs(x.get("combined_score", 0)),
            reverse=True,
        )[:self.max_positions]

        return {
            "total_pairs_analyzed": len(signals),
            "buy_count": len(buy_signals),
            "sell_count": len(sell_signals),
            "hold_count": len(hold_signals),
            "top_opportunities": best_opportunities,
            "market_bias": "BULLISH" if len(buy_signals) > len(sell_signals) else
                          "BEARISH" if len(sell_signals) > len(buy_signals) else "NEUTRAL",
        }
