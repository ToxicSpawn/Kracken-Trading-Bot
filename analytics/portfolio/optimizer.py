from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PortfolioOptimConfig:
    max_weight_per_symbol: float = 0.4
    risk_aversion: float = 2.0  # >1 => more conservative than pure Kelly


class PortfolioOptimizer:
    """
    Simple Kelly-like portfolio weight allocator.

    Given per-symbol:
      - edge (expected return per trade)
      - variance (risk proxy)
    it returns normalized weights that respect max_weight_per_symbol
    and risk_aversion.
    """

    def __init__(self, cfg: PortfolioOptimConfig | None = None) -> None:
        self.cfg = cfg or PortfolioOptimConfig()

    def optimize(
        self,
        edges: Dict[str, float],
        variances: Dict[str, float],
    ) -> Dict[str, float]:
        raw_weights: Dict[str, float] = {}

        for sym, edge in edges.items():
            var = variances.get(sym, 0.0)
            if var <= 0 or edge <= 0:
                continue

            kelly = edge / var
            kelly /= self.cfg.risk_aversion
            w = max(0.0, min(self.cfg.max_weight_per_symbol, float(kelly)))
            raw_weights[sym] = w

        total = sum(raw_weights.values())
        if total <= 0:
            return {sym: 0.0 for sym in edges.keys()}

        return {sym: w / total for sym, w in raw_weights.items()}
