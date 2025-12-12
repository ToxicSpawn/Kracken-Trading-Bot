from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class CorrelationConfig:
    lookback: int = 200
    hedge_threshold: float = 0.8   # corr above this triggers hedge consideration
    min_points: int = 50           # minimum overlapping points needed


class CorrelationEngine:
    """
    Computes correlation between symbols and proposes simple hedge relationships.

    Usage:
        engine = CorrelationEngine()
        corr = engine.compute_matrix(price_series)
        proposals = engine.propose_hedges(exposures, corr)
    """

    def __init__(self, cfg: CorrelationConfig | None = None) -> None:
        self.cfg = cfg or CorrelationConfig()

    def compute_matrix(
        self,
        price_series: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        aligned = []
        for sym, series in price_series.items():
            if len(series) < self.cfg.lookback:
                continue
            aligned.append(series.tail(self.cfg.lookback).rename(sym))

        if not aligned:
            return pd.DataFrame()

        df = pd.concat(aligned, axis=1).dropna()
        if len(df) < self.cfg.min_points:
            return pd.DataFrame()

        returns = df.pct_change().dropna()
        return returns.corr()

    def propose_hedges(
        self,
        exposures: Dict[str, float],  # symbol -> notional exposure
        corr_matrix: pd.DataFrame,
    ) -> List[Tuple[str, str, float]]:
        """
        Returns list of (source_symbol, hedge_symbol, hedge_fraction):
          - hedge_fraction is fraction of source exposure to hedge *in opposite direction*
        """
        if corr_matrix.empty:
            return []

        proposals: List[Tuple[str, str, float]] = []
        symbols = list(exposures.keys())

        for i, s1 in enumerate(symbols):
            exp1 = exposures.get(s1, 0.0)
            if abs(exp1) < 1e-6:
                continue
            for s2 in symbols[i + 1 :]:
                exp2 = exposures.get(s2, 0.0)
                if abs(exp2) < 1e-6:
                    continue
                try:
                    c = float(corr_matrix.loc[s1, s2])
                except Exception:
                    continue
                if np.isnan(c):
                    continue
                if abs(c) >= self.cfg.hedge_threshold:
                    hedge_frac = 0.5 * abs(c)  # hedge half of exposure scaled by correlation
                    proposals.append((s1, s2, hedge_frac))

        return proposals
