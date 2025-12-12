from __future__ import annotations
import logging
from typing import Dict, Optional

from exchange.venues import Venue
from exchange.kraken_client import KrakenClient
from exchange.binance_client import BinanceClient
from exchange.coinbase_client import CoinbaseClient
from exchange.venue_symbol_mapper import VenueSymbolMapper

logger = logging.getLogger(__name__)


class MultiVenueOrderExecutor:
    """
    Multi-venue execution engine (v1,000,000)
    Auto-symbol mapping + notional → base conversion.

    Works for:
      • Kraken (spot AUD)
      • Binance spot/futures (USDT)
      • Coinbase spot (USD)
    """

    def __init__(self):
        self.kraken = KrakenClient()
        self.binance = BinanceClient()
        self.coinbase = CoinbaseClient()
        self.mapper = VenueSymbolMapper()

    async def execute(
        self,
        venue: Venue,
        symbol: str,
        side: str,
        notional_quote: float,
        futures: bool = False,
        fallback: Optional[Venue] = None,
    ) -> Dict:

        mapped_symbol = self.mapper.map_symbol(symbol, venue, futures=futures)

        try:
            if venue == Venue.KRAKEN:
                return await self._exec_kraken(mapped_symbol, side, notional_quote)
            if venue == Venue.BINANCE:
                return await self._exec_binance(mapped_symbol, side, notional_quote, futures)
            if venue == Venue.COINBASE:
                return await self._exec_coinbase(mapped_symbol, side, notional_quote)

            return {"ok": False, "venue": None, "message": f"Unknown venue {venue}"}

        except Exception as exc:
            logger.error("Execution failed on %s: %s", venue.value, exc)

            if fallback:
                return await self.execute(
                    fallback, symbol, side, notional_quote, futures=(fallback == Venue.BINANCE)
                )

            return {"ok": False, "venue": venue.value, "message": str(exc)}

    # ------------------------ Venue-specific -----------------------------

    async def _exec_kraken(self, symbol: str, side: str, notional: float):
        t = await self.kraken.fetch_ticker(symbol)
        px = float(t["last"])
        amt = notional / px
        out = await self.kraken.create_order(symbol, side, amt)
        return {"ok": True, "venue": "kraken", "order_id": out.get("id"), "fill_price": px}

    async def _exec_binance(self, symbol: str, side: str, notional: float, futures: bool):
        t = await self.binance.fetch_ticker(symbol, futures)
        px = float(t["last"])
        amt = notional / px
        out = await self.binance.create_order(symbol, side, amt, futures)
        return {"ok": True, "venue": "binance", "order_id": out.get("id"), "fill_price": px}

    async def _exec_coinbase(self, symbol: str, side: str, notional: float):
        t = await self.coinbase.fetch_ticker(symbol)
        px = float(t["last"])
        amt = notional / px
        out = await self.coinbase.create_order(symbol, side, amt)
        return {"ok": True, "venue": "coinbase", "order_id": out.get("id"), "fill_price": px}
