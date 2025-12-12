from __future__ import annotations

import logging

from agents.base import BaseAgent

logger = logging.getLogger(__name__)


class PortfolioAgent(BaseAgent):
    """Portfolio agent (placeholder for future PnL calc)."""

    async def step(self) -> None:
        for name, acc in self.ctx.state.accounts.items():
            logger.debug("PortfolioAgent: account=%s equity=%.2f balance=%.2f", name, acc.equity, acc.balance)
