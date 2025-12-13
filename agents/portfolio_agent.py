from __future__ import annotations

import logging
from datetime import datetime

from agents.base import BaseAgent

logger = logging.getLogger(__name__)


class PortfolioAgent(BaseAgent):
    """Portfolio agent - tracks positions and records trade outcomes for loss cluster supervisor."""

    def __init__(self, name, ctx, interval: float = 30.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self._previous_positions: dict[str, dict[str, float]] = {}  # account -> symbol -> size
        self._previous_realized_pnl: dict[str, float] = {}  # account -> realized_pnl

    async def step(self) -> None:
        for name, acc in self.ctx.state.accounts.items():
            logger.debug("PortfolioAgent: account=%s equity=%.2f balance=%.2f", name, acc.equity, acc.balance)

            # Track position closes and realized PnL changes for loss cluster supervisor
            prev_positions = self._previous_positions.get(name, {})
            prev_pnl = self._previous_realized_pnl.get(name, acc.realized_pnl)

            # Check for position closes (size went to 0)
            for symbol, pos in acc.positions.items():
                prev_size = prev_positions.get(symbol, 0.0)
                if prev_size != 0.0 and pos.size == 0.0:
                    # Position closed - calculate realized PnL for this trade
                    trade_pnl = pos.realized_pnl
                    if trade_pnl != 0.0:
                        self.ctx.state.meta["last_trade_pnl"] = float(trade_pnl)
                        self.ctx.state.meta["last_trade_ts"] = datetime.utcnow().isoformat()
                        logger.info(
                            "PortfolioAgent: Position closed %s on %s, realized_pnl=%.2f",
                            symbol,
                            name,
                            trade_pnl,
                        )

            # Also check for realized PnL changes (when PnL is updated even without position size change)
            if acc.realized_pnl != prev_pnl:
                delta_pnl = acc.realized_pnl - prev_pnl
                if delta_pnl != 0.0:
                    self.ctx.state.meta["last_trade_pnl"] = float(delta_pnl)
                    self.ctx.state.meta["last_trade_ts"] = datetime.utcnow().isoformat()
                    logger.debug(
                        "PortfolioAgent: Realized PnL changed on %s, delta=%.2f",
                        name,
                        delta_pnl,
                    )

            # Update tracking
            self._previous_positions[name] = {sym: pos.size for sym, pos in acc.positions.items()}
            self._previous_realized_pnl[name] = acc.realized_pnl
