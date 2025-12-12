from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class AlphaStats:
    mean: float
    std: float
    sharpe: float
    win_rate: float
    num: int


class AlphaRanker:
    """
    v120000 alpha ranking helper.

    Feed recent returns per strategy and get stats; use this in MetaController
    or OmegaBrain to update strategy_weights.
    """

    def compute(self, returns: Sequence[float]) -> AlphaStats:
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return AlphaStats(mean=0.0, std=0.0, sharpe=0.0, win_rate=0.0, num=0)

        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        sharpe = float(mean / (std + 1e-9))
        win_rate = float((arr > 0).mean())
        return AlphaStats(mean=mean, std=std, sharpe=sharpe, win_rate=win_rate, num=len(arr))
