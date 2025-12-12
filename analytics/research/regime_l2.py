from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class RegimeL2:
    label: str
    trend: float
    vol: float
    vol_state: str
    shock: bool


class RegimeL2Engine:
    """
    v120000 higher-level regime classifier.

    Input: close prices & returns
    Output: label + shock flag + vol state.
    """

    def classify(self, prices: List[float]) -> RegimeL2:
        if len(prices) < 60:
            return RegimeL2(label="unknown", trend=0.0, vol=0.0, vol_state="unknown", shock=False)

        s = pd.Series(prices, dtype="float64")
        rets = s.pct_change().dropna()

        x = np.arange(len(s))
        if s.std() == 0:
            trend_corr = 0.0
        else:
            trend_corr = float(np.corrcoef(x, s.values)[0, 1])

        vol = float(rets.std())
        if vol < 0.01:
            vol_state = "calm"
        elif vol < 0.03:
            vol_state = "normal"
        elif vol < 0.06:
            vol_state = "elevated"
        else:
            vol_state = "chaotic"

        shock = vol_state == "chaotic"

        if trend_corr > 0.3:
            label = f"up_{vol_state}"
        elif trend_corr < -0.3:
            label = f"down_{vol_state}"
        else:
            label = f"sideways_{vol_state}"

        return RegimeL2(label=label, trend=trend_corr, vol=vol, vol_state=vol_state, shock=shock)
