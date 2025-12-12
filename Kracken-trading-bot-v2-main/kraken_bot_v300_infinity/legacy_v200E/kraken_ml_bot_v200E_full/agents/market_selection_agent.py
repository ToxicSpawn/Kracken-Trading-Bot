from __future__ import annotations

import logging
from typing import Dict

from agents.base import AgentContext, BaseAgent

logger = logging.getLogger(__name__)


class MarketSelectionAgent(BaseAgent):
    """
    v120000 Market Selection Agent.

    - Ranks pairs by recent edge (win_rate, pnl)
    - Limits active pairs to a max universe (e.g. top N)
    - Writes into state.meta["pair_performance"] and lets Omega/Router decide

    You can later extend this to use volatility, volume, and correlations.
    """

    def __init__(self, name: str, ctx: AgentContext, interval: float = 600.0, max_active: int = 8) -> None:
        super().__init__(name, ctx, interval=interval)
        self.max_active = max_active

    async def step(self) -> None:
        state = self.ctx.state

        perf: Dict[str, Dict[str, float]] = state.meta.get("pair_performance", {})

        if not perf:
            return

        ranked = sorted(
            perf.items(),
            key=lambda kv: (kv[1].get("pnl", 0.0), kv[1].get("win_rate", 0.5)),
            reverse=True,
        )

        active = [sym for sym, _stats in ranked[: self.max_active]]
        inactive = [sym for sym, _stats in ranked[self.max_active :]]

        state.meta["market_selection"] = {
            "active": active,
            "inactive": inactive,
        }

        logger.info(
            "MarketSelectionAgent: active=%s inactive=%s",
            active,
            inactive,
        )
