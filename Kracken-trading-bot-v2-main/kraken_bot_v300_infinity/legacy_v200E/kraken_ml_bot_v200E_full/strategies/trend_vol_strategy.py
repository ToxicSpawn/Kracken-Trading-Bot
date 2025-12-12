from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


class TrendVolStrategy:
    """Compute trend + volatility scores with regime awareness.

    Trend is derived from the slope of a rolling linear regression and
    centered around 0.5 (neutral). Volatility is scaled using a
    percentile-like transformation to better distinguish "calm" vs
    "stormy" regimes without overreacting to single spikes.
    """


    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def score(self, ohlcv: List[list]) -> tuple[float, float]:
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
            return 0.0, 0.0

        returns = close.pct_change().dropna()
        if returns.empty:
            return 0.0, 0.0

        # Linear regression slope normalised by price to gauge trend
        # strength and direction. Centered at 0.5 (neutral) for
        # compatibility with existing consumers.
        x = np.arange(len(close))
        slope, _ = np.polyfit(x, close, 1)
        slope_norm = slope / close.iloc[-1] if close.iloc[-1] else 0.0
        trend_score = float(np.clip((np.tanh(slope_norm * 500) + 1) / 2, 0.0, 1.0))

        # Use rolling volatility but dampen extremes with tanh so sudden
        # spikes do not dominate the sizing logic.
        vol_raw = float(returns.rolling(window=20).std().iloc[-1])
        vol_score = float(np.clip(np.tanh(vol_raw * 80), 0.0, 1.0))

        return trend_score, vol_score
