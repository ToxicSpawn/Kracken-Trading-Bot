from __future__ import annotations

import logging
from typing import List

from exchange.kraken_client import KrakenClient

logger = logging.getLogger(__name__)



class DataFeed:
    def __init__(self) -> None:
        self.client = KrakenClient()

    async def get_recent_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> List[list]:
        try:
            return await self.client.fetch_ohlcv(symbol, timeframe, limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DataFeed error for %s: %s", symbol, exc)
            return []
