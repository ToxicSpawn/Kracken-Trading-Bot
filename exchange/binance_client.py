from __future__ import annotations
import os
import ccxt  # type: ignore


class BinanceClient:
    """
    Binance Client (Spot + USDT Futures)
    v1,000,000 Edition

    Env vars:
      BINANCE_API_KEY
      BINANCE_API_SECRET
    """

    def __init__(self) -> None:
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        self.spot = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

        self.futures = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

    async def fetch_ticker(self, symbol: str, futures: bool = False):
        client = self.futures if futures else self.spot
        return client.fetch_ticker(symbol)

    async def create_order(self, symbol: str, side: str, amount: float, futures: bool = False):
        client = self.futures if futures else self.spot
        return client.create_order(symbol=symbol, type="market", side=side.lower(), amount=amount)
