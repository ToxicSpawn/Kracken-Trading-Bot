from __future__ import annotations

from enum import Enum
from typing import List

import numpy as np
import pandas as pd


class RegimeLabel(str, Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"


class RegimeClassifier:
    """Classify the current market regime using rolling returns."""

    def classify(self, ohlcv: List[list]) -> RegimeLabel:
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        closes = df["close"].astype(float)
        if len(closes) < 50:
            return RegimeLabel.SIDEWAYS

        returns = closes.pct_change().dropna()
        recent = returns.tail(50)

        momentum = float(np.tanh((closes.iloc[-1] - closes.iloc[-50]) / closes.iloc[-50]))
        volatility = float(recent.std())
        auto_corr = float(recent.autocorr(lag=1)) if len(recent) > 1 else 0.0

        if volatility > 0.025:
            return RegimeLabel.HIGH_VOL
        if momentum > 0.02 and auto_corr > 0:
            return RegimeLabel.TRENDING
        if momentum < -0.02 and auto_corr > 0:
            return RegimeLabel.TRENDING
        if auto_corr < -0.1:
            return RegimeLabel.MEAN_REVERTING
        return RegimeLabel.SIDEWAYS
