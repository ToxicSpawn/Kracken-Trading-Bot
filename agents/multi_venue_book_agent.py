from __future__ import annotations

import logging
from time import perf_counter
from typing import Dict

from agents.base import BaseAgent
from exchange.kraken_client import KrakenClient
from exchange.venues import Venue

logger = logging.getLogger(__name__)


class MultiVenueBookAgent(BaseAgent):
    """
    v1,000,000 Multi-Venue Order Book Agent.

    - Fetches best bid/ask + depth per venue for each symbol.
    - Measures rough latency per request.
    - Writes snapshots into state.meta["venue_book"][symbol][venue] = {...}
    """

    def __init__(self, name: str, ctx, symbols, interval: float = 3.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self.symbols = symbols
        self.kraken = KrakenClient()
        # self.binance = BinanceClient(...)
        # self.coinbase = CoinbaseClient(...)

    async def step(self) -> None:  # type: ignore[override]
        venue_book: Dict[str, Dict[str, dict]] = self.ctx.state.meta.setdefault("venue_book", {})

        # For now, Kraken only – you can add others later.
        for symbol in self.symbols:
            t0 = perf_counter()
            try:
                ob = await self.kraken.fetch_order_book(symbol, limit=10)
                latency_ms = (perf_counter() - t0) * 1000.0
            except Exception as exc:  # noqa: BLE001
                logger.warning("MultiVenueBookAgent: failed order book for %s on Kraken: %s", symbol, exc)
                continue

            bids = ob.get("bids") if isinstance(ob, dict) else None
            asks = ob.get("asks") if isinstance(ob, dict) else None
            if not bids or not asks:
                continue

            best_bid, bid_size = bids[0]
            best_ask, ask_size = asks[0]
            mid = (best_bid + best_ask) / 2.0
            spread_bps = (best_ask - best_bid) / mid * 10_000.0

            sym_map = venue_book.setdefault(symbol, {})
            sym_map[Venue.KRAKEN.value] = {
                "best_bid": float(best_bid),
                "best_ask": float(best_ask),
                "bid_size_quote": float(best_bid * bid_size),
                "ask_size_quote": float(best_ask * ask_size),
                "mid_price": float(mid),
                "spread_bps": float(spread_bps),
                "latency_ms": float(latency_ms),
            }

            logger.debug(
                "MultiVenueBookAgent: %s Kraken mid=%.4f spread=%.2fbps depth_bid=%.0f depth_ask=%.0f lat=%.1fms",
                symbol,
                mid,
                spread_bps,
                best_bid * bid_size,
                best_ask * ask_size,
                latency_ms,
            )
