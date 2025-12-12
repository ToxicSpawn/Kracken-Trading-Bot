from __future__ import annotations

import logging
from agents.base import BaseAgent

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    """
    RiskAgent — v120000 Omega Edition

    Adds:
    - Omega "OFF" and "SHOCK" enforcement
    - Volatility state checks (Regime-L2)
    - Data Quality enforcement (flags > 10)
    - Central policy + Omega joint override
    """

    async def step(self) -> None:
        decision = self.ctx.policy.evaluate(self.ctx.state)

        router = self.ctx.state.meta.get("router", {})
        omega = self.ctx.state.meta.get("omega", {})

        regime_l2 = (self.ctx.state.meta.get("regime_l2") or {})
        vol_state = str(regime_l2.get("vol_state", "normal"))
        shock = bool(regime_l2.get("shock", False))

        dq_flags = int(self.ctx.state.meta.get("data_quality_flags", 0))

        # Policy reasons stack
        if router.get("mode") == "OFF":
            decision.reasons.append("Router mode OFF")

        if omega.get("mode") == "OFF":
            decision.reasons.append("Omega mode OFF")

        if shock:
            decision.reasons.append("Regime-L2 shock mode")

        if vol_state == "chaotic":
            decision.reasons.append("Market volatility chaotic")

        if dq_flags > 10:
            decision.reasons.append("Data-quality failure")

        # Apply decision
        self.ctx.state.meta["policy_reasons"] = decision.reasons

        # Enforce global halt
        if not decision.can_trade and self.ctx.state.trading_enabled:
            logger.warning(
                "RiskAgent: disabling trading (%s)",
                "; ".join(decision.reasons)
            )
            self.ctx.state.trading_enabled = False
