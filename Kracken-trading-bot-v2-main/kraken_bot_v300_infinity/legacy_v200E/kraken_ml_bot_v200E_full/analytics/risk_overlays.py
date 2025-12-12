from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StressResult:
    size_multiplier: float
    reason: str | None = None


class StressTester:
    """Translate microstructure stress into conservative sizing."""

    def __init__(self, max_slippage_bps: float = 75.0, max_volatility: float = 0.03) -> None:
        self.max_slippage_bps = max_slippage_bps
        self.max_volatility = max_volatility

    def evaluate(self, volatility: float, slippage_bps: float) -> StressResult:
        stress = 0.0
        reason = None

        if volatility > self.max_volatility:
            stress += (volatility - self.max_volatility) / self.max_volatility
            reason = "high_volatility_stress"
        if slippage_bps > self.max_slippage_bps:
            stress += (slippage_bps - self.max_slippage_bps) / self.max_slippage_bps
            reason = "slippage_stress" if reason is None else f"{reason}+slippage"

        if stress == 0:
            return StressResult(size_multiplier=1.0)

        size_multiplier = max(0.25, 1.0 - min(stress, 1.5) * 0.4)
        return StressResult(size_multiplier=size_multiplier, reason=reason)
