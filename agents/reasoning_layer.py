from __future__ import annotations

import logging

from agents.base import AgentContext, BaseAgent
from core.omega_brain import OmegaBrain

logger = logging.getLogger(__name__)


class ReasoningLayerAgent(BaseAgent):
    """
    v120000 Reasoning Layer Agent.

    - Calls OmegaBrain.decide()
    - Normalizes decisions
    - Writes to state.meta["omega"]
    - Optionally adjusts existing router decisions
    """

    def __init__(self, name: str, ctx: AgentContext, interval: float = 30.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self.brain = OmegaBrain()

    async def step(self) -> None:
        state = self.ctx.state
        decision = self.brain.decide(state)

        state.meta["omega"] = {
            "mode": decision.mode.value,
            "risk_multiplier": decision.risk_multiplier,
            "allowed_pairs": decision.allowed_pairs,
            "blocked_pairs": decision.blocked_pairs,
            "strategy_weights": decision.strategy_weights,
            "notes": decision.notes,
        }

        logger.info(
            "ReasoningLayer: mode=%s risk=%.2f allowed=%s notes=%s",
            decision.mode.value,
            decision.risk_multiplier,
            decision.allowed_pairs,
            "; ".join(decision.notes),
        )

        router = state.meta.get("router", {})
        router["mode"] = decision.mode.value
        router["enabled_pairs"] = decision.allowed_pairs
        router["disabled_pairs"] = decision.blocked_pairs
        router["strategy_weights"] = decision.strategy_weights
        router["risk_multiplier"] = decision.risk_multiplier
        state.meta["router"] = router
