from __future__ import annotations

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
        self._client = ccxt.kraken({
            "apiKey": os.getenv("KRAKEN_API_KEY"),
            "secret": os.getenv("KRAKEN_API_SECRET"),
            "enableRateLimit": True,
        })

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        return self._client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    async def fetch_order_book(self, symbol: str, limit: int = 10):
        return self._client.fetch_order_book(symbol, limit=limit)

    async def fetch_ticker(self, symbol: str):
        return self._client.fetch_ticker(symbol)

    async def create_order(self, symbol: str, side: str, amount: float):
        side = side.lower()
        return self._client.create_order(symbol, "market", side, amount)
