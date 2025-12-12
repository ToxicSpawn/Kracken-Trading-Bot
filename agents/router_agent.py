from __future__ import annotations

import asyncio
import logging

from agents.base import AgentContext, BaseAgent
from core.alpha_router import GlobalAlphaRouter

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """
    v90000 RouterAgent.

    Periodically runs GlobalAlphaRouter and writes decisions into GlobalState.meta:

      state.meta["router"] = {
          "mode": "...",
          "enabled_pairs": [...],
          "disabled_pairs": [...],
          "strategy_weights": {...},
          "risk_multiplier": x,
      }
    """

    def __init__(self, name: str, ctx: AgentContext, interval: float = 60.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self.router = GlobalAlphaRouter()

    async def run_loop(self) -> None:  # type: ignore[override]
        logger.info("RouterAgent %s loop started", self.name)
        while self._running:
            try:
                decision = self.router.decide(self.ctx.state)

                self.ctx.state.meta["router"] = {
                    "mode": decision.mode.value,
                    "enabled_pairs": decision.enabled_pairs,
                    "disabled_pairs": decision.disabled_pairs,
                    "strategy_weights": decision.strategy_weights,
                    "risk_multiplier": decision.risk_multiplier,
                }

                logger.info(
                    "RouterAgent: mode=%s risk_mult=%.2f enabled=%s notes=%s",
                    decision.mode.value,
                    decision.risk_multiplier,
                    decision.enabled_pairs,
                    "; ".join(decision.notes),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("RouterAgent error: %s", exc)

            await asyncio.sleep(self.interval)
