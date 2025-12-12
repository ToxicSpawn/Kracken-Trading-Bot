from __future__ import annotations

import logging
from typing import Dict, List

from agents.base import BaseAgent
from analytics.research.cross_venue_arb import CrossVenueArbEngine
from exchange.multi_venue_executor import MultiVenueOrderExecutor
from exchange.venues import Venue, DEFAULT_VENUES

logger = logging.getLogger(__name__)


class ArbitrageAgent(BaseAgent):
    """
    v2,000,000 ArbitrageAgent

    - Scans cross-venue spreads using venue_book.
    - Only trades when net_spread_bps > threshold.
    - Size is small, hedged, and Omega-aware.

    IMPORTANT:
    Start with tiny size and/or DRY_RUN mode.
    """

    def __init__(
        self,
        name: str,
        ctx,
        symbols: List[str],
        interval: float = 15.0,
        min_net_spread_bps: float = 20.0,
        notional_per_trade: float = 10.0,
    ) -> None:
        super().__init__(name, ctx, interval=interval)
        self.symbols = symbols
        self.min_net_spread_bps = min_net_spread_bps
        self.notional_per_trade = notional_per_trade

        fee_map: Dict[Venue, float] = {
            v: cfg.fee_bps for v, cfg in DEFAULT_VENUES.items()
        }
        self.engine = CrossVenueArbEngine(fee_bps=fee_map)
        self.executor = MultiVenueOrderExecutor()

    async def step(self) -> None:
        state = self.ctx.state
        omega = state.meta.get("omega", {})
        mode = omega.get("mode", "NORMAL")

        # Only allow arb in NORMAL / AGGRESSIVE for now
        if mode in ("OFF", "SAFE", "SHOCK"):
            return

        venue_book = state.meta.get("venue_book", {})
        if not venue_book:
            return

        for sym in self.symbols:
            best = self.engine.find_best_spread(sym, venue_book)
            if not best:
                continue

            if best.net_spread_bps < self.min_net_spread_bps:
                continue

            logger.info(
                "ArbAgent: %s buy %s @ %.4f sell %s @ %.4f net=%.1fbps",
                sym,
                best.leg_buy_venue.value,
                best.buy_price,
                best.leg_sell_venue.value,
                best.sell_price,
                best.net_spread_bps,
            )

            # For now: symmetric notional on both legs in quote terms.
            # In real life: manage inventory & limits per venue carefully.
            await self._execute_pair_trade(best)

    async def _execute_pair_trade(self, snap) -> None:
        # BUY on cheap venue
        res_buy = await self.executor.execute(
            venue=snap.leg_buy_venue,
            symbol=snap.base_symbol,     # mapped inside executor
            side="BUY",
            notional_quote=self.notional_per_trade,
            futures=(snap.leg_buy_venue == Venue.BINANCE),
        )

        # SELL on rich venue
        res_sell = await self.executor.execute(
            venue=snap.leg_sell_venue,
            symbol=snap.base_symbol,
            side="SELL",
            notional_quote=self.notional_per_trade,
            futures=(snap.leg_sell_venue == Venue.BINANCE),
        )

        logger.info(
            "ArbAgent: executed legs → buy_ok=%s sell_ok=%s",
            res_buy.get("ok"), res_sell.get("ok"),
        )
