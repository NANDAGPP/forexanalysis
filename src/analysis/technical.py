"""
Technical Analysis Engine.
Computes indicators and generates trading signals based on real-time market data.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Computes technical indicators and generates market signals."""

    def __init__(self):
        self.indicators_config = {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2,
            "ema_short": 9,
            "ema_medium": 21,
            "ema_long": 50,
            "atr_period": 14,
            "stoch_k": 14,
            "stoch_d": 3,
            "adx_period": 14,
        }

    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators on OHLCV dataframe."""
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for indicator computation")
            return df

        result = df.copy()

        result["rsi"] = ta.rsi(result["close"], length=self.indicators_config["rsi_period"])

        macd = ta.macd(
            result["close"],
            fast=self.indicators_config["macd_fast"],
            slow=self.indicators_config["macd_slow"],
            signal=self.indicators_config["macd_signal"],
        )
        if macd is not None:
            result = pd.concat([result, macd], axis=1)

        bb = ta.bbands(
            result["close"],
            length=self.indicators_config["bb_period"],
            std=self.indicators_config["bb_std"],
        )
        if bb is not None:
            result = pd.concat([result, bb], axis=1)

        result["ema_9"] = ta.ema(result["close"], length=self.indicators_config["ema_short"])
        result["ema_21"] = ta.ema(result["close"], length=self.indicators_config["ema_medium"])
        result["ema_50"] = ta.ema(result["close"], length=self.indicators_config["ema_long"])
        result["sma_200"] = ta.sma(result["close"], length=200)

        result["atr"] = ta.atr(
            result["high"],
            result["low"],
            result["close"],
            length=self.indicators_config["atr_period"],
        )

        stoch = ta.stoch(
            result["high"],
            result["low"],
            result["close"],
            k=self.indicators_config["stoch_k"],
            d=self.indicators_config["stoch_d"],
        )
        if stoch is not None:
            result = pd.concat([result, stoch], axis=1)

        adx = ta.adx(
            result["high"],
            result["low"],
            result["close"],
            length=self.indicators_config["adx_period"],
        )
        if adx is not None:
            result = pd.concat([result, adx], axis=1)

        result["obv"] = ta.obv(result["close"], result["volume"])

        ichimoku = ta.ichimoku(result["high"], result["low"], result["close"])
        if ichimoku is not None and len(ichimoku) == 2:
            result = pd.concat([result, ichimoku[0]], axis=1)

        result["pivot"] = (result["high"] + result["low"] + result["close"]) / 3
        result["support_1"] = 2 * result["pivot"] - result["high"]
        result["resistance_1"] = 2 * result["pivot"] - result["low"]
        result["support_2"] = result["pivot"] - (result["high"] - result["low"])
        result["resistance_2"] = result["pivot"] + (result["high"] - result["low"])

        return result

    def get_signal_strength(self, df: pd.DataFrame) -> dict:
        """
        Analyze indicators and return a comprehensive signal assessment.
        Returns signal strength from -100 (strong sell) to +100 (strong buy).
        """
        if df.empty or len(df) < 50:
            return {"signal": "NEUTRAL", "strength": 0, "details": {}}

        latest = df.iloc[-1]
        signals = []
        details = {}

        if "rsi" in df.columns and not pd.isna(latest.get("rsi")):
            rsi = latest["rsi"]
            if rsi < 30:
                signals.append(("RSI", 30, "Oversold - Buy signal"))
            elif rsi > 70:
                signals.append(("RSI", -30, "Overbought - Sell signal"))
            elif rsi < 45:
                signals.append(("RSI", 10, "Slightly bullish"))
            elif rsi > 55:
                signals.append(("RSI", -10, "Slightly bearish"))
            else:
                signals.append(("RSI", 0, "Neutral"))
            details["rsi"] = {"value": float(rsi), "signal": signals[-1][2]}

        macd_col = [c for c in df.columns if "MACD_" in c and "MACDs" not in c and "MACDh" not in c]
        macd_signal_col = [c for c in df.columns if "MACDs_" in c]
        macd_hist_col = [c for c in df.columns if "MACDh_" in c]

        if macd_col and macd_signal_col:
            macd_val = latest[macd_col[0]]
            macd_sig = latest[macd_signal_col[0]]
            if not pd.isna(macd_val) and not pd.isna(macd_sig):
                if macd_val > macd_sig:
                    signals.append(("MACD", 25, "Bullish crossover"))
                else:
                    signals.append(("MACD", -25, "Bearish crossover"))
                details["macd"] = {
                    "value": float(macd_val),
                    "signal_line": float(macd_sig),
                    "signal": signals[-1][2],
                }

        if "ema_9" in df.columns and "ema_21" in df.columns:
            ema9 = latest.get("ema_9")
            ema21 = latest.get("ema_21")
            ema50 = latest.get("ema_50")
            price = latest["close"]

            if not pd.isna(ema9) and not pd.isna(ema21):
                if price > ema9 > ema21:
                    signals.append(("EMA", 30, "Strong uptrend"))
                elif price < ema9 < ema21:
                    signals.append(("EMA", -30, "Strong downtrend"))
                elif ema9 > ema21:
                    signals.append(("EMA", 15, "Mild uptrend"))
                else:
                    signals.append(("EMA", -15, "Mild downtrend"))
                details["ema"] = {
                    "ema_9": float(ema9),
                    "ema_21": float(ema21),
                    "signal": signals[-1][2],
                }

        bb_upper_col = [c for c in df.columns if "BBU_" in c]
        bb_lower_col = [c for c in df.columns if "BBL_" in c]
        bb_mid_col = [c for c in df.columns if "BBM_" in c]

        if bb_upper_col and bb_lower_col:
            bb_upper = latest[bb_upper_col[0]]
            bb_lower = latest[bb_lower_col[0]]
            price = latest["close"]
            if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                bb_range = bb_upper - bb_lower
                if bb_range > 0:
                    bb_position = (price - bb_lower) / bb_range
                    if bb_position > 0.95:
                        signals.append(("Bollinger", -20, "Near upper band - Potential reversal"))
                    elif bb_position < 0.05:
                        signals.append(("Bollinger", 20, "Near lower band - Potential bounce"))
                    else:
                        signals.append(("Bollinger", 0, "Within bands"))
                    details["bollinger"] = {
                        "position": float(bb_position),
                        "upper": float(bb_upper),
                        "lower": float(bb_lower),
                        "signal": signals[-1][2],
                    }

        stoch_k_col = [c for c in df.columns if "STOCHk_" in c]
        stoch_d_col = [c for c in df.columns if "STOCHd_" in c]
        if stoch_k_col and stoch_d_col:
            stoch_k = latest[stoch_k_col[0]]
            stoch_d = latest[stoch_d_col[0]]
            if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                if stoch_k < 20 and stoch_d < 20:
                    signals.append(("Stochastic", 20, "Oversold"))
                elif stoch_k > 80 and stoch_d > 80:
                    signals.append(("Stochastic", -20, "Overbought"))
                else:
                    signals.append(("Stochastic", 0, "Neutral"))
                details["stochastic"] = {
                    "k": float(stoch_k),
                    "d": float(stoch_d),
                    "signal": signals[-1][2],
                }

        adx_col = [c for c in df.columns if c.startswith("ADX_")]
        if adx_col:
            adx_val = latest[adx_col[0]]
            if not pd.isna(adx_val):
                trend_strength = "Weak" if adx_val < 25 else "Strong" if adx_val > 50 else "Moderate"
                details["adx"] = {"value": float(adx_val), "trend_strength": trend_strength}

        if not signals:
            return {"signal": "NEUTRAL", "strength": 0, "details": details}

        total_strength = sum(s[1] for s in signals)
        avg_strength = total_strength / len(signals)
        normalized = max(-100, min(100, avg_strength * 2))

        if normalized > 30:
            signal = "STRONG BUY"
        elif normalized > 10:
            signal = "BUY"
        elif normalized < -30:
            signal = "STRONG SELL"
        elif normalized < -10:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        return {
            "signal": signal,
            "strength": round(normalized, 2),
            "details": details,
            "individual_signals": [
                {"indicator": s[0], "score": s[1], "description": s[2]} for s in signals
            ],
        }

    def get_support_resistance(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        """Identify key support and resistance levels."""
        if df.empty or len(df) < lookback:
            return {"support": [], "resistance": []}

        recent = df.tail(lookback)
        price = recent["close"].iloc[-1]

        highs = recent["high"].values
        lows = recent["low"].values

        resistance_levels = []
        support_levels = []

        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and \
               highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                if highs[i] > price:
                    resistance_levels.append(float(highs[i]))

            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and \
               lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                if lows[i] < price:
                    support_levels.append(float(lows[i]))

        resistance_levels = sorted(set(resistance_levels))[:5]
        support_levels = sorted(set(support_levels), reverse=True)[:5]

        return {
            "current_price": float(price),
            "resistance": resistance_levels,
            "support": support_levels,
        }

    def get_market_summary(self, df: pd.DataFrame, pair: str) -> dict:
        """Generate a comprehensive market summary for a pair."""
        indicators_df = self.compute_all_indicators(df)
        signal_data = self.get_signal_strength(indicators_df)
        sr_levels = self.get_support_resistance(indicators_df)

        latest = indicators_df.iloc[-1] if not indicators_df.empty else {}
        prev = indicators_df.iloc[-2] if len(indicators_df) > 1 else latest

        price_change = float(latest["close"] - prev["close"]) if not indicators_df.empty else 0
        price_change_pct = (price_change / prev["close"] * 100) if not indicators_df.empty and prev["close"] != 0 else 0

        return {
            "pair": pair,
            "current_price": float(latest["close"]) if not indicators_df.empty else None,
            "price_change": round(price_change, 5),
            "price_change_pct": round(price_change_pct, 4),
            "signal": signal_data,
            "support_resistance": sr_levels,
            "recommendation": self._generate_recommendation(signal_data, sr_levels),
        }

    def _generate_recommendation(self, signal_data: dict, sr_levels: dict) -> dict:
        """Generate a trading recommendation based on analysis."""
        strength = signal_data.get("strength", 0)
        signal = signal_data.get("signal", "NEUTRAL")

        if abs(strength) < 10:
            action = "HOLD"
            confidence = "LOW"
            reason = "Mixed signals - no clear direction"
        elif strength > 30:
            action = "BUY"
            confidence = "HIGH"
            reason = "Multiple indicators align bullish"
        elif strength > 10:
            action = "BUY"
            confidence = "MEDIUM"
            reason = "Moderate bullish bias"
        elif strength < -30:
            action = "SELL"
            confidence = "HIGH"
            reason = "Multiple indicators align bearish"
        elif strength < -10:
            action = "SELL"
            confidence = "MEDIUM"
            reason = "Moderate bearish bias"
        else:
            action = "HOLD"
            confidence = "LOW"
            reason = "Insufficient signal strength"

        price = sr_levels.get("current_price", 0)
        stop_loss = None
        take_profit = None

        if action == "BUY" and sr_levels.get("support"):
            stop_loss = sr_levels["support"][0]
            if sr_levels.get("resistance"):
                take_profit = sr_levels["resistance"][0]
        elif action == "SELL" and sr_levels.get("resistance"):
            stop_loss = sr_levels["resistance"][0]
            if sr_levels.get("support"):
                take_profit = sr_levels["support"][0]

        return {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": round(
                abs(take_profit - price) / abs(price - stop_loss), 2
            ) if stop_loss and take_profit and abs(price - stop_loss) > 0 else None,
        }
