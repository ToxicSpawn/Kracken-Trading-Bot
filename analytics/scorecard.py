from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List

from core.state import GlobalState


@dataclass
class BotScorecard:
    score: float
    rating: str
    reasons: List[str]

    def as_dict(self) -> dict:
        return {"score": self.score, "rating": self.rating, "reasons": self.reasons}


def _clamp_score(score: float) -> float:
    return max(0.0, min(20.0, score))


def compute_scorecard(state: GlobalState) -> BotScorecard:
    """Return a composite health score for the bot (0-20)."""

    score = 20.0
    reasons: List[str] = []

    if not state.trading_enabled:
        score -= 5.0
        reasons.append("Trading paused by policy guardrails")

    total_pnl = state.total_realized_pnl + state.total_unrealized_pnl
    if total_pnl < 0:
        penalty = min(8.0, sqrt(abs(total_pnl)))
        score -= penalty
        reasons.append(f"Negative PnL headwind: {total_pnl:.2f}")

    dq = state.meta.get("data_quality", {})
    dq_flags = sum(1 for frames in dq.values() for status in frames.values() if status.get("gap") or status.get("stale"))
    if dq_flags:
        score -= min(4.0, dq_flags * 1.0)
        reasons.append("Data-quality alerts detected")

    policy_reasons = state.meta.get("policy_reasons", [])
    if policy_reasons:
        score -= 2.0
        reasons.append("Risk policy constraints active")

    riskiness = max((acc.risk_multiplier for acc in state.accounts.values()), default=1.0)
    if riskiness > 1.0:
        score -= min(3.0, (riskiness - 1.0) * 2.0)
        reasons.append(f"Risk multiplier elevated to {riskiness:.2f}")

    score = _clamp_score(score)
    rating = "20/10" if score >= 18 else "excellent" if score >= 15 else "stable" if score >= 10 else "needs_attention"

    if not reasons:
        reasons.append("All systems nominal")

    return BotScorecard(score=score, rating=rating, reasons=reasons)
