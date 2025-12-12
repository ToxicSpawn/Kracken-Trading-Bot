from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.state import GlobalState


@dataclass
class PolicyConfig:
    max_daily_loss_aud: float
    absolute_max_drawdown_pct: float
    max_single_position_pct: float
    per_symbol_max_loss_aud: float = -150.0
    allowed_trading_hours_utc: Optional[List[int]] = field(default=None)
    max_volatility_score: float = 0.95
    event_blackout_hours_utc: Optional[List[int]] = field(default=None)
    min_liquidity_score: float = 0.15
    max_slippage_bps: float = 120.0
    regime_blocklist: Optional[List[str]] = field(default_factory=list)
    rolling_loss_window_min: float = 0.0
    rolling_loss_limit_aud: float = 0.0


@dataclass
class PolicyDecision:
    can_trade: bool
    reasons: List[str] = field(default_factory=list)


class PolicyEngine:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        self._equity_peaks: Dict[str, float] = {}

    @classmethod
    def from_yaml(cls, yaml_cfg: Dict[str, Any]) -> "PolicyEngine":
        pol = yaml_cfg.get("policies", {})
        return cls(
            PolicyConfig(
                max_daily_loss_aud=float(os.getenv("DAILY_MAX_LOSS_AUD", pol.get("max_daily_loss_aud", -200.0))),
                absolute_max_drawdown_pct=float(pol.get("absolute_max_drawdown_pct", -20.0)),
                max_single_position_pct=float(pol.get("max_single_position_pct", 0.4)),
                per_symbol_max_loss_aud=float(pol.get("per_symbol_max_loss_aud", -150.0)),
                allowed_trading_hours_utc=pol.get("allowed_trading_hours_utc"),
                max_volatility_score=float(pol.get("max_volatility_score", 0.95)),
                event_blackout_hours_utc=pol.get("event_blackout_hours_utc"),
                min_liquidity_score=float(pol.get("min_liquidity_score", 0.15)),
                max_slippage_bps=float(pol.get("max_slippage_bps", 120.0)),
                regime_blocklist=pol.get("regime_blocklist", []),
                rolling_loss_window_min=float(pol.get("rolling_loss_window_min", 0.0)),
                rolling_loss_limit_aud=float(pol.get("rolling_loss_limit_aud", 0.0)),
            )
        )

    def _update_equity_peak(self, account_name: str, equity: float) -> float:
        peak = self._equity_peaks.get(account_name, equity)
        if equity > peak:
            peak = equity
            self._equity_peaks[account_name] = equity
        else:
            self._equity_peaks[account_name] = peak
        return peak

    def evaluate(self, state: GlobalState) -> PolicyDecision:
        reasons: List[str] = []

        if state.total_realized_pnl < self.config.max_daily_loss_aud:
            reasons.append(
                f"daily PnL {state.total_realized_pnl:.2f} breached max loss {self.config.max_daily_loss_aud:.2f}"
            )

        for acct_name, acct in state.accounts.items():
            peak = self._update_equity_peak(acct_name, acct.equity)
            if peak > 0:
                drawdown_pct = ((acct.equity - peak) / peak) * 100
                if drawdown_pct < self.config.absolute_max_drawdown_pct:
                    reasons.append(
                        f"account {acct_name} drawdown {drawdown_pct:.2f}% below limit {self.config.absolute_max_drawdown_pct:.2f}%"
                    )

            for sym, pos in acct.positions.items():
                if pos.realized_pnl < self.config.per_symbol_max_loss_aud:
                    reasons.append(
                        f"{sym} PnL {pos.realized_pnl:.2f} breached per-symbol loss {self.config.per_symbol_max_loss_aud:.2f}"
                    )

        if self.config.allowed_trading_hours_utc is not None:
            current_hour = datetime.utcnow().hour
            if current_hour not in self.config.allowed_trading_hours_utc:
                reasons.append(
                    f"trading hour {current_hour} not in allowed hours {self.config.allowed_trading_hours_utc} (UTC)"
                )

        if self.config.event_blackout_hours_utc:
            current_hour = datetime.utcnow().hour
            if current_hour in self.config.event_blackout_hours_utc:
                reasons.append(
                    f"event blackout hour {current_hour} (UTC) in {self.config.event_blackout_hours_utc}"
                )

        for sym, ps in state.pairs.items():
            if ps.vol_score > self.config.max_volatility_score:
                reasons.append(
                    f"{sym} vol_score {ps.vol_score:.2f} above max {self.config.max_volatility_score:.2f}"
                )
            if ps.liquidity_score < self.config.min_liquidity_score:
                reasons.append(
                    f"{sym} liquidity {ps.liquidity_score:.2f} below min {self.config.min_liquidity_score:.2f}"
                )
            if ps.slippage_bps > self.config.max_slippage_bps:
                reasons.append(
                    f"{sym} slippage {ps.slippage_bps:.1f}bps exceeds cap {self.config.max_slippage_bps:.1f}bps"
                )
            if self.config.regime_blocklist and ps.regime in self.config.regime_blocklist:
                reasons.append(f"{sym} regime {ps.regime} blocked")

        return PolicyDecision(can_trade=not reasons, reasons=reasons)

    def can_trade(self, state: GlobalState) -> bool:
        return self.evaluate(state).can_trade
