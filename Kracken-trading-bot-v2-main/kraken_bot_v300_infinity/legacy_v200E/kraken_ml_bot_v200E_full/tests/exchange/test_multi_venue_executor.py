import asyncio

import pytest

from exchange import multi_venue_executor as mve
from exchange.venues import Venue


class _StubClient:
    async def fetch_ticker(self, symbol: str):
        return {"last": 1.0}

    async def create_order(self, symbol: str, side: str, amount: float):
        return {"id": "stub"}


async def _run_guarded_exec(monkeypatch):
    monkeypatch.setattr(mve, "KrakenClient", lambda: _StubClient())
    monkeypatch.setattr(mve, "BinanceClient", lambda: _StubClient())
    monkeypatch.setattr(mve, "CoinbaseClient", lambda: _StubClient())

    monkeypatch.setattr(
        mve.DEFAULT_VENUES[Venue.KRAKEN],
        "max_notional_per_order",
        10.0,
    )

    executor = mve.MultiVenueOrderExecutor()
    return await executor.execute(Venue.KRAKEN, "BTC/AUD", "buy", 20.0)


def test_execute_rejects_notional_above_cap(monkeypatch):
    result = asyncio.run(_run_guarded_exec(monkeypatch))

    assert result["ok"] is False
    assert "exceeds per-order cap" in result["message"]

