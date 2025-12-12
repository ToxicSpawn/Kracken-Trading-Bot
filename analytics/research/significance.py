from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import ttest_1samp


@dataclass
class SignificanceResult:
    mean: float
    t_stat: float
    p_value: float
    significant: bool


class SignificanceTester:
    """
    v120000 simple statistical significance tester.

    Use on per-trade returns, per-strategy returns, etc.

    Example:
        tester = SignificanceTester(alpha=0.05)
        res = tester.test(returns)
        if res.significant and res.mean > 0:
            # strategy has statistically significant positive edge
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def test(self, series: Sequence[float]) -> SignificanceResult:
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 20:
            return SignificanceResult(mean=0.0, t_stat=0.0, p_value=1.0, significant=False)

        mean = float(arr.mean())
        t_stat, p_value = ttest_1samp(arr, 0.0)
        significant = p_value < self.alpha

        return SignificanceResult(mean=mean, t_stat=float(t_stat), p_value=float(p_value), significant=bool(significant))
