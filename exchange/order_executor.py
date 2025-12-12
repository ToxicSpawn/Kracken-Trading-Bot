from __future__ import annotations

import logging
import os

from exchange.kraken_client import KrakenClient
from core.global_state import get_global_state

logger = logging.getLogger(__name__)



class OrderExecutor:
    def __init__(self) -> None:
        self.client = KrakenClient()

    async def execute(self, symbol: str, side: str, notional_aud: float):
        mode = os.getenv("BOT_MODE", "paper")
        logger.info("OrderExecutor: %s %s notional=%.2f mode=%s", side, symbol, notional_aud, mode)

        if mode != "live":
            logger.info("OrderExecutor: paper mode → no real order placed.")
            return

        state = get_global_state()
        pair = state.pairs.get(symbol)
        if not pair or not pair.last_price:
            logger.warning("OrderExecutor: no last_price for %s; cannot size order", symbol)
            return

        amount = notional_aud / pair.last_price

        try:
            await self.client.create_order(symbol=symbol, side=side.lower(), amount=amount)
        except Exception as exc:  # noqa: BLE001
            logger.error("OrderExecutor: error placing order: %s", exc)

    async def set_target_positions(self, targets: dict[str, float]) -> None:
        """Set target base position sizes per symbol.

        The executor will calculate order deltas against current positions when running
        in live mode. In paper mode we simply log the intended targets.
        """

        mode = os.getenv("BOT_MODE", "paper")
        logger.info("OrderExecutor: setting targets %s mode=%s", targets, mode)

        if mode != "live":
            logger.info("OrderExecutor: paper mode → no real order placed.")
            return

        state = get_global_state()
        account = next(iter(state.accounts.values()), None)

        for symbol, target_size in targets.items():
            current_size = 0.0
            if account:
                current_position = account.positions.get(symbol)
                if current_position:
                    current_size = current_position.size

            delta = target_size - current_size
            if abs(delta) < 1e-9:
                logger.info("OrderExecutor: %s already at target %.6f", symbol, target_size)
                continue

            side = "buy" if delta > 0 else "sell"

            try:
                await self.client.create_order(symbol=symbol, side=side, amount=abs(delta))
            except Exception as exc:  # noqa: BLE001
                logger.error("OrderExecutor: error placing target order for %s: %s", symbol, exc)
