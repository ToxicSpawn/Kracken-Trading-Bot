from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from .state import GlobalState


class OmegaMode(str, Enum):
    OFF = "OFF"  # Hard stop – don't trade
    SAFE = "SAFE"  # Tiny risk, core markets only
    NORMAL = "NORMAL"  # Default mode
    AGGRESSIVE = "AGGRESSIVE"  # Up risk, more markets
    ROTATE = "ROTATE"  # Market rotation / exploration
    SHOCK = "SHOCK"  # Hedge/defensive only


@dataclass
class OmegaDecision:
    mode: OmegaMode
    risk_multiplier: float
    allowed_pairs: List[str]
    blocked_pairs: List[str]
    strategy_weights: Dict[str, float]
    notes: List[str]


class OmegaBrain:
    """
    v120000 Omega AI global brain.

    Reads:
      - state.total_realized_pnl / unrealized
      - state.meta["drawdown_pct"]
      - state.meta["avg_market_vol"]
      - state.meta["pnl_volatility"]
      - state.meta["pair_performance"]
      - state.meta["data_quality_flags"]
      - state.meta["regime_l2"] (if provided by advanced regime engine)

    Writes decisions used by:
      - RouterAgent
      - ExecutionAgent
      - RiskAgent
      - MarketSelectionAgent
    """

    def decide(self, state: GlobalState) -> OmegaDecision:
        notes: List[str] = []

        total_pnl = state.total_realized_pnl + state.total_unrealized_pnl
        dd_pct = float(state.meta.get("drawdown_pct", 0.0))
        avg_vol = float(state.meta.get("avg_market_vol", 0.0))
        pnl_vol = float(state.meta.get("pnl_volatility", 0.0))
        dq_flags = int(state.meta.get("data_quality_flags", 0))
        regime_l2: Dict[str, Any] = state.meta.get("regime_l2", {})

        deep_dd = dd_pct <= -30.0
        med_dd = -30.0 < dd_pct <= -15.0
        high_vol = avg_vol > 0.05 or pnl_vol > 0.03

        regime_name = str(regime_l2.get("label", "unknown"))
        shock_flag = bool(regime_l2.get("shock", False))

        # --- Choose global mode ---
        if dq_flags > 10:
            mode = OmegaMode.OFF
            notes.append("Omega: too many data-quality issues -> OFF")
        elif shock_flag:
            mode = OmegaMode.SHOCK
            notes.append(f"Omega: regime '{regime_name}' flagged shock -> SHOCK")
        elif deep_dd:
            mode = OmegaMode.SAFE
            notes.append(f"Omega: deep drawdown {dd_pct:.1f}% -> SAFE")
        elif med_dd and high_vol:
            mode = OmegaMode.SAFE
            notes.append("Omega: medium DD + high vol -> SAFE")
        elif total_pnl > 0 and not high_vol and dd_pct > -10.0:
            mode = OmegaMode.AGGRESSIVE
            notes.append("Omega: profitable, low vol, low DD -> AGGRESSIVE")
        else:
            mode = OmegaMode.NORMAL
            notes.append("Omega: default -> NORMAL")

        # --- Risk multiplier ---
        if mode == OmegaMode.OFF:
            risk_mult = 0.0
        elif mode == OmegaMode.SAFE:
            risk_mult = 0.4
        elif mode == OmegaMode.NORMAL:
            risk_mult = 1.0
        elif mode == OmegaMode.AGGRESSIVE:
            risk_mult = 1.6
        elif mode == OmegaMode.ROTATE:
            risk_mult = 0.8
        else:  # SHOCK
            risk_mult = 0.3

        # --- Per-pair enable/disable ---
        perf = state.meta.get("pair_performance", {})  # symbol -> {pnl, win_rate}
        allowed_pairs: List[str] = []
        blocked_pairs: List[str] = []

        for sym, stats in perf.items():
            win_rate = float(stats.get("win_rate", 0.5))
            pair_pnl = float(stats.get("pnl", 0.0))

            # In SHOCK mode, only keep pairs that are strongly profitable
            if mode == OmegaMode.SHOCK and (win_rate < 0.55 or pair_pnl <= 0):
                blocked_pairs.append(sym)
                notes.append(f"Omega: shock mode - disabling {sym}")
                continue

            if win_rate < 0.4 and pair_pnl < 0:
                blocked_pairs.append(sym)
                notes.append(
                    f"Omega: disabling {sym} (weak edge wr={win_rate:.2f}, pnl={pair_pnl:.2f})"
                )
            else:
                allowed_pairs.append(sym)

        if not allowed_pairs and perf:
            core = max(perf.items(), key=lambda kv: kv[1].get("win_rate", 0.5))[0]
            allowed_pairs.append(core)
            notes.append(f"Omega: forcing {core} enabled as fallback")

        # --- Strategy weighting ---
        weights: Dict[str, float] = {
            "rsi": 0.3,
            "trend": 0.2,
            "ml": 0.5,
        }

        if mode == OmegaMode.SAFE:
            weights = {"rsi": 0.45, "trend": 0.35, "ml": 0.20}
        elif mode == OmegaMode.AGGRESSIVE:
            weights = {"rsi": 0.15, "trend": 0.25, "ml": 0.60}
        elif mode == OmegaMode.SHOCK:
            weights = {"rsi": 0.6, "trend": 0.4, "ml": 0.0}

        # Optionally: if research found good params, slightly favor ML/Trend
        research = state.meta.get("research", {})
        best_score = float(research.get("best_score", 0.0))
        if best_score > 0:
            # Tiny nudge towards ML if research is positive
            weights["ml"] = min(0.7, weights.get("ml", 0.5) + 0.1)
            weights["trend"] = min(0.3, weights.get("trend", 0.2) + 0.05)

        return OmegaDecision(
            mode=mode,
            risk_multiplier=risk_mult,
            allowed_pairs=allowed_pairs,
            blocked_pairs=blocked_pairs,
            strategy_weights=weights,
            notes=notes,
        )
