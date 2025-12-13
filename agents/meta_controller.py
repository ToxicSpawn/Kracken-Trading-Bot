from __future__ import annotations

import logging
from agents.base import BaseAgent
from analytics.scorecard import compute_scorecard
from core.live_metrics import set_system_state
from metrics.server import update_metrics

logger = logging.getLogger(__name__)


class MetaController(BaseAgent):
    """
    MetaController — v120000 Omega Edition

    Duties:
    - Reports Omega mode + risk multiplier
    - Tracks scorecard + regime-l2 + pair-performance
    - Updates Prometheus metrics with Omega fields
    """

    async def step(self) -> None:
        state = self.ctx.state
        omega = state.meta.get("omega", {})
        router = state.meta.get("router", {})

        scorecard = compute_scorecard(state)
        state.meta["scorecard"] = scorecard.as_dict()

        # Extract useful Omega metrics
        omega_mode = omega.get("mode", "UNKNOWN")
        risk_mult = float(omega.get("risk_multiplier", 1.0))
        allowed = omega.get("allowed_pairs", [])
        blocked = omega.get("blocked_pairs", [])

        regime_l2 = state.meta.get("regime_l2") or {}
        vol_state = str(regime_l2.get("vol_state", "unknown"))
        shock = bool(regime_l2.get("shock", False))

        logger.info(
            "MetaController: Ωmode=%s risk=%.2f trading=%s pnl=%.2f vol=%s shock=%s",
            omega_mode,
            risk_mult,
            state.trading_enabled,
            state.total_realized_pnl,
            vol_state,
            shock,
        )

        # Extend Prometheus metrics
        scorecard.extra["omega_mode"] = omega_mode
        scorecard.extra["omega_risk_multiplier"] = risk_mult
        scorecard.extra["omega_allowed"] = len(allowed)
        scorecard.extra["omega_blocked"] = len(blocked)
        scorecard.extra["regime_l2_vol_state"] = vol_state
        scorecard.extra["regime_l2_shock"] = int(shock)

        update_metrics(scorecard)

        # Update live metrics system state
        set_system_state(state)
