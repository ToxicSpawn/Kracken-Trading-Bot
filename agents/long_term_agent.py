from __future__ import annotations

import logging

from agents.base import BaseAgent
from strategies.rsi_ml_strategy import RsiMlStrategy
from exchange.data_feed import DataFeed

logger = logging.getLogger(__name__)


class LongTermAgent(BaseAgent):
    """Long-term signals (1h RSI-based)."""

    def __init__(self, name, ctx, symbols, interval: float = 60.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self.symbols = symbols
        self.feed = DataFeed()

    async def step(self) -> None:
        for symbol in self.symbols:
            ohlcv = await self.feed.get_recent_ohlcv(symbol, timeframe="1h", limit=120)
            if not ohlcv:
                continue
            strat = RsiMlStrategy(symbol)
            sig = strat.generate_signal(ohlcv)
            ps = self.ctx.state.pairs.setdefault(symbol, None)
            if ps is not None:
                ps.last_signal_long = sig.signal.name
            logger.info("LongTerm: %s → %s (conf=%.2f)", symbol, sig.signal.name, sig.confidence)
