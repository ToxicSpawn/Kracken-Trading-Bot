from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .venue_models import VenueBookSnapshot, VenueScore, score_venue


@dataclass
class MicrostructureProfile:
    liquidity_score: float
    slippage_bps: float
    volatility: float
    spread_proxy_bps: float


class MicrostructureProfiler:
    """Lightweight liquidity/impact model derived from OHLCV candles."""

    def __init__(self, depth_sensitivity: float = 1.2) -> None:
        self.depth_sensitivity = depth_sensitivity

    def evaluate(self, ohlcv: List[list]) -> MicrostructureProfile:
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        closes = df["close"].astype(float)
        volumes = df["volume"].astype(float)

        if len(df) < 20:
            return MicrostructureProfile(
                liquidity_score=0.3,
                slippage_bps=25.0,
                volatility=0.0,
                spread_proxy_bps=25.0,
            )

        window = min(len(df), 120)
        closes = closes.tail(window)
        volumes = volumes.tail(window)

        returns = closes.pct_change().dropna()
        volatility = float(returns.std())
        range_pct = (df["high"].tail(window) - df["low"].tail(window)) / closes
        spread_proxy_bps = float(np.clip(range_pct.mean() * 10_000, 1.0, 80.0))

        vol_score = float(np.clip(np.tanh(volatility * 50), 0.0, 1.0))
        depth = float(volumes.rolling(window=12, min_periods=5).mean().iloc[-1])
        depth_norm = float(np.clip(np.tanh(depth / (volumes.mean() + 1e-9)), 0.0, 1.0))

        liquidity_score = float(np.clip((depth_norm * 0.6) + (1 - vol_score) * 0.4, 0.0, 1.0))

        impact_penalty = (1 - liquidity_score) ** self.depth_sensitivity
        slippage_bps = float(np.clip(spread_proxy_bps * (1 + impact_penalty), 1.0, 150.0))

        return MicrostructureProfile(
            liquidity_score=liquidity_score,
            slippage_bps=slippage_bps,
            volatility=volatility,
            spread_proxy_bps=spread_proxy_bps,
        )


__all__ = [
    "MicrostructureProfile",
    "MicrostructureProfiler",
    "VenueBookSnapshot",
    "VenueScore",
    "score_venue",
]
