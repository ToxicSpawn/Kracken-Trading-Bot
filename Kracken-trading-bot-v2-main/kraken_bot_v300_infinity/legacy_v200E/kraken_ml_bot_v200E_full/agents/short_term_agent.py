from __future__ import annotations

import logging

from agents.base import BaseAgent
from strategies.rsi_ml_strategy import RsiMlStrategy
from strategies.trend_vol_strategy import TrendVolStrategy
from exchange.data_feed import DataFeed

logger = logging.getLogger(__name__)


class ShortTermAgent(BaseAgent):
    """Short-term intraday signals (5m RSI + trend/vol scores)."""

    def __init__(self, name, ctx, symbols, interval: float = 5.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self.symbols = symbols
        self.feed = DataFeed()

    async def step(self) -> None:
        for symbol in self.symbols:
            ohlcv = await self.feed.get_recent_ohlcv(symbol, timeframe="5m", limit=120)
            if not ohlcv:
                continue
            rsi_strat = RsiMlStrategy(symbol)
            sig = rsi_strat.generate_signal(ohlcv)

            tv = TrendVolStrategy(symbol)
            trend_score, vol_score = tv.score(ohlcv)

            ps = self.ctx.state.pairs.setdefault(symbol, None)
            if ps is not None:
                ps.last_signal_short = sig.signal.name
                ps.trend_score = trend_score
                ps.vol_score = vol_score
            logger.info(
                "ShortTerm: %s → %s (conf=%.2f, trend=%.2f, vol=%.2f)",
                symbol,
                sig.signal.name,
                sig.confidence,
                trend_score,
                vol_score,
            )
