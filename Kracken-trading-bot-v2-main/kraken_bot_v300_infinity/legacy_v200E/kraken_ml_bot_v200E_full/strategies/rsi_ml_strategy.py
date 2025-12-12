from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from strategies.base import StrategySignal, SignalType


class RsiMlStrategy:
    """Signal generator that blends classic momentum indicators.

    v9000: parameters are tunable so the research pipeline can grid-search
    RSI and MACD settings.
    """

    def __init__(
        self,
        symbol: str,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ) -> None:
        self.symbol = symbol
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        window = self.rsi_window
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _compute_macd_hist(self, close: pd.Series) -> pd.Series:
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        return macd - signal

    def generate_signal(self, ohlcv: List[list]) -> StrategySignal:
        df = pd.DataFrame(
            ohlcv,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )
        close = df["close"].astype(float)

        if len(close) < 30:
            return StrategySignal(symbol=self.symbol, signal=SignalType.HOLD, confidence=0.2)

        rsi = self._compute_rsi(close)
        latest_rsi = rsi.iloc[-1]
        macd_hist = self._compute_macd_hist(close).iloc[-1]

        rolling = close.rolling(window=20)
        mid = rolling.mean().iloc[-1]
        std = rolling.std(ddof=0).iloc[-1]
        if np.isnan(mid) or std == 0:
            bb_pos = 0.0
        else:
            bb_pos = float(np.clip((close.iloc[-1] - mid) / (2 * std), -1.5, 1.5))

        rsi_bias = -((latest_rsi - 50.0) / 50.0)
        macd_bias = float(np.tanh(macd_hist * 5))
        bb_bias = float(np.clip(-bb_pos, -1.0, 1.0))

        if np.isnan(rsi_bias):
            rsi_bias = 0.0
        if np.isnan(macd_bias):
            macd_bias = 0.0
        if np.isnan(bb_bias):
            bb_bias = 0.0

        momentum = float(np.tanh(close.pct_change().rolling(window=5).mean().iloc[-1] * 50))

        bias = (0.6 * rsi_bias) + (0.2 * macd_bias) + (0.15 * bb_bias) + (0.05 * momentum)

        if bias > 0.1:
            sig = SignalType.BUY
        elif bias < -0.1:
            sig = SignalType.SELL
        else:
            sig = SignalType.HOLD

        vola = close.pct_change().rolling(window=20).std().iloc[-1]
        vola = 0.0 if np.isnan(vola) else vola
        vola_penalty = float(np.clip(np.tanh(vola * 40), 0.0, 0.4))
        confidence = float(np.clip(abs(bias) + (0.2 - vola_penalty), 0.2, 0.95))

        return StrategySignal(symbol=self.symbol, signal=sig, confidence=confidence)
