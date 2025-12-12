from __future__ import annotations
import os
import ccxt  # type: ignore


class CoinbaseClient:
    """
    Coinbase Spot Client
    v1,000,000 Edition

    Env vars:
      COINBASE_API_KEY
      COINBASE_API_SECRET
      COINBASE_API_PASSPHRASE
    """

    def __init__(self) -> None:
        self.client = ccxt.coinbase({
            "apiKey": os.getenv("COINBASE_API_KEY"),
            "secret": os.getenv("COINBASE_API_SECRET"),
            "password": os.getenv("COINBASE_API_PASSPHRASE"),
            "enableRateLimit": True,
        })

    async def fetch_ticker(self, symbol: str):
        return self.client.fetch_ticker(symbol)

    async def create_order(self, symbol: str, side: str, amount: float):
        return self.client.create_order(symbol=symbol, type="market", side=side.lower(), amount=amount)
