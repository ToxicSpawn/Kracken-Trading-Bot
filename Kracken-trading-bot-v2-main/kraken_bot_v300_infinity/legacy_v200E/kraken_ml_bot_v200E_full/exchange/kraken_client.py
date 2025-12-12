from __future__ import annotations

import asyncio
import os

import ccxt  # type: ignore


class KrakenClient:
    """
    Kraken spot client via ccxt.

    Env vars:
      KRAKEN_API_KEY
      KRAKEN_API_SECRET
    """

    def __init__(self) -> None:
        self._client = ccxt.kraken(
            {
                "apiKey": os.getenv("KRAKEN_API_KEY"),
                "secret": os.getenv("KRAKEN_API_SECRET"),
                "enableRateLimit": True,
            }
        )

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        # Pass limit as keyword to avoid shifting the `since` argument positionally.
        return await asyncio.to_thread(self._client.fetch_ohlcv, symbol, timeframe, None, limit)

    async def fetch_order_book(self, symbol: str, limit: int = 10):
        return await asyncio.to_thread(self._client.fetch_order_book, symbol, limit)

    async def fetch_ticker(self, symbol: str):
        return await asyncio.to_thread(self._client.fetch_ticker, symbol)

    async def create_order(self, symbol: str, side: str, amount: float):
        side = side.lower()
        return await asyncio.to_thread(self._client.create_order, symbol, "market", side, amount)
