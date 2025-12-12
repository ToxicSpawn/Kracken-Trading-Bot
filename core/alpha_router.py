from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from core.state import GlobalState


class Mode(str, Enum):
    OFF = "OFF"  # no trading
    SAFE = "SAFE"  # tiny size, few pairs, strict risk
    NORMAL = "NORMAL"  # default
    AGGRESSIVE = "AGGRESSIVE"  # higher size, more pairs
    SHOCK = "SHOCK"  # weird mode: hedged, defensive, maybe only hedge/scalp


@dataclass
class RouterDecision:
    mode: Mode
    enabled_pairs: List[str]
    disabled_pairs: List[str]
    strategy_weights: Dict[str, float]  # e.g. {"rsi":0.3, "ml":0.5, "trend":0.2}
    risk_multiplier: float  # global risk multiplier
    notes: List[str]


class GlobalAlphaRouter:
    """
    v90000 Global Alpha Router.

    Reads:
      - PnL, drawdown, risk metrics
      - volatility, correlation, microstructure labels
      - strategy hit-rates / recent performance (if tracked in state.meta)

    Outputs:
      - global mode (OFF/SAFE/NORMAL/AGGRESSIVE/SHOCK)
      - which pairs should be enabled/disabled
      - weights for each strategy family (RSI, ML, TrendVol, etc.)
      - a global risk multiplier

    Other agents (RiskAgent, ExecutionAgent, HedgeAgent, MLAgent) read these
    decisions from GlobalState.meta and adapt behavior.
    """

    def decide(self, state: GlobalState) -> RouterDecision:
        notes: List[str] = []

        total_pnl = state.total_realized_pnl + state.total_unrealized_pnl
        dd_pct = float(state.meta.get("drawdown_pct", 0.0))
        pnl_vol = float(state.meta.get("pnl_volatility", 0.0))
        avg_vol = float(state.meta.get("avg_market_vol", 0.0))  # you can set this from MarketDataAgent
        dq_issues = int(state.meta.get("data_quality_flags", 0))

        # basic thresholds - tune later
        deep_dd = dd_pct <= -30
        medium_dd = -30 < dd_pct <= -15
        high_vol = avg_vol > 0.05 or pnl_vol > 0.03

        # choose mode
        if dq_issues > 10:
            mode = Mode.OFF
            notes.append("Too many data quality issues")
        elif deep_dd:
            mode = Mode.SAFE
            notes.append(f"Deep drawdown {dd_pct:.1f}% -> SAFE mode")
        elif medium_dd and high_vol:
            mode = Mode.SAFE
            notes.append("Medium drawdown + high vol -> SAFE mode")
        elif total_pnl > 0 and not high_vol and dd_pct > -10:
            mode = Mode.AGGRESSIVE
            notes.append("PnL positive, low vol, low DD -> AGGRESSIVE mode")
        else:
            mode = Mode.NORMAL
            notes.append("Default mode NORMAL")

        # risk multiplier by mode
        if mode == Mode.OFF:
            risk_mult = 0.0
        elif mode == Mode.SAFE:
            risk_mult = 0.5
        elif mode == Mode.NORMAL:
            risk_mult = 1.0
        elif mode == Mode.AGGRESSIVE:
            risk_mult = 1.5
        else:  # SHOCK
            risk_mult = 0.3

        # which pairs to favor or drop
        perf = state.meta.get("pair_performance", {})  # e.g. {"SOL/AUD": {"pnl": x, "win_rate": y}, ...}
        enabled_pairs: List[str] = []
        disabled_pairs: List[str] = []

        for sym, stats in perf.items():
            win_rate = float(stats.get("win_rate", 0.5))
            pair_pnl = float(stats.get("pnl", 0.0))

            if mode in (Mode.OFF,):
                disabled_pairs.append(sym)
                continue

            if win_rate < 0.4 and pair_pnl < 0:
                disabled_pairs.append(sym)
                notes.append(f"Disabling {sym} (weak edge: wr={win_rate:.2f}, pnl={pair_pnl:.2f})")
            else:
                enabled_pairs.append(sym)

        # if nothing enabled, keep at least one core symbol
        if not enabled_pairs and perf:
            core = max(perf.items(), key=lambda kv: kv[1].get("win_rate", 0.5))[0]
            enabled_pairs.append(core)
            notes.append(f"Forcing {core} enabled as fallback")

        # simple strategy weights (can be made smarter later)
        strategy_weights: Dict[str, float] = {
            "rsi": 0.3,
            "trend": 0.2,
            "ml": 0.5,
        }

        if mode == Mode.SAFE:
            strategy_weights["ml"] = 0.3
            strategy_weights["rsi"] = 0.4
            strategy_weights["trend"] = 0.3
        elif mode == Mode.AGGRESSIVE:
            strategy_weights["ml"] = 0.6
            strategy_weights["trend"] = 0.25
            strategy_weights["rsi"] = 0.15

        return RouterDecision(
            mode=mode,
            enabled_pairs=enabled_pairs,
            disabled_pairs=disabled_pairs,
            strategy_weights=strategy_weights,
            risk_multiplier=risk_mult,
            notes=notes,
        )
