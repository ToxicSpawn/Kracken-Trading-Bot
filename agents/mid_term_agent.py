from __future__ import annotations

import logging

from agents.base import BaseAgent
from strategies.rsi_ml_strategy import RsiMlStrategy
from exchange.data_feed import DataFeed

logger = logging.getLogger(__name__)


class MidTermAgent(BaseAgent):
    """Mid-term swing signals (15m RSI-based)."""

    def __init__(self, name, ctx, symbols, interval: float = 15.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self.symbols = symbols
        self.feed = DataFeed()

    async def step(self) -> None:
        for symbol in self.symbols:
            ohlcv = await self.feed.get_recent_ohlcv(symbol, timeframe="15m", limit=120)
            if not ohlcv:
                continue
            strat = RsiMlStrategy(symbol)
            sig = strat.generate_signal(ohlcv)
            ps = self.ctx.state.pairs.setdefault(symbol, None)
            if ps is not None:
                ps.last_signal_mid = sig.signal.name
            logger.info("MidTerm: %s → %s (conf=%.2f)", symbol, sig.signal.name, sig.confidence)
