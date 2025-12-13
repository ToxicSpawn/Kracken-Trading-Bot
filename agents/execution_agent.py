from __future__ import annotations

import logging
import os

from analytics import RegimeLabel
from analytics.rl import RlPositionSizer, RlSizerConfig
from agents.base import BaseAgent
from core.state import PairSnapshot
from core.live_metrics import on_order_submitted, on_order_error, order_latency_timer
from notifications.telegram import send_telegram_message
from exchange.venue_router import VenueRouter
from exchange.multi_venue_executor import MultiVenueOrderExecutor
from exchange.venues import Venue

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    """Execution agent (v200E).

    - Blends short/mid/long signals.
    - Uses trend/vol scores to scale position size.
    - Respects global trading_enabled flag.
    - In 'paper' mode: logs only.
    - In 'live' mode: routes orders via MultiVenueOrderExecutor.
    """

    def __init__(self, name, ctx, interval: float = 10.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self.rlsizer = RlPositionSizer(RlSizerConfig())
        self.venue_router = VenueRouter()
        self.multi_executor = MultiVenueOrderExecutor()

    async def step(self) -> None:
        if not self.ctx.state.trading_enabled:
            logger.debug("ExecutionAgent: trading disabled; skipping.")
            return

        router = self.ctx.state.meta.get("router", {})
        omega = self.ctx.state.meta.get("omega", {})

        mode = omega.get("mode", router.get("mode", "NORMAL"))
        strat_weights = omega.get(
            "strategy_weights", router.get("strategy_weights", {"rsi": 0.3, "trend": 0.2, "ml": 0.5})
        )
        global_risk_mult = float(router.get("risk_multiplier", 1.0)) * float(omega.get("risk_multiplier", 1.0))
        # Apply loss cluster throttle multiplier if present
        loss_cluster_throttle = float(self.ctx.state.meta.get("loss_cluster_throttle_multiplier", 1.0))
        global_risk_mult *= loss_cluster_throttle
        enabled_pairs = set(
            omega.get("allowed_pairs")
            or router.get("enabled_pairs")
            or self.ctx.state.pairs.keys()
        )

        if mode == "OFF":
            logger.info("ExecutionAgent: Omega mode OFF - skipping trades.")
            return

        target_positions: dict[str, tuple[float, str]] = {}

        for symbol, ps in list(self.ctx.state.pairs.items()):
            if symbol not in enabled_pairs:
                logger.debug("ExecutionAgent: %s disabled by router; skipping.", symbol)
                continue

            blended = self._blend_signals(ps, strat_weights, mode)
            if blended in (None, "HOLD"):
                continue

            size_mult = self._volatility_size_multiplier(ps)
            size_mult *= ps.stress_multiplier
            if ps.liquidity_score < 0.3 or ps.slippage_bps > 90:
                logger.info(
                    "ExecutionAgent: skipping %s due to poor liquidity (liq=%.2f, slip=%.1f bps)",
                    symbol,
                    ps.liquidity_score,
                    ps.slippage_bps,
                )
                await send_telegram_message(
                    f"[Guardrail] Skipping {symbol}: liquidity={ps.liquidity_score:.2f}, slip={ps.slippage_bps:.1f}bps"
                )
                continue

            regime_mult = self._regime_sizing(ps)
            size_mult *= regime_mult

            base_notional = 1.0
            if mode == "SAFE":
                base_notional *= 0.5
            elif mode == "AGGRESSIVE":
                base_notional *= 1.2

            notional_aud = await self._compute_notional_for_symbol(ps, base_notional, size_mult)
            notional_aud *= global_risk_mult
            rl_mult = notional_aud / max(base_notional * size_mult, 1e-9)

            logger.info(
                "ExecutionAgent: blended %s for %s, trend=%.2f vol=%.2f size_mult=%.2f regime=%s stress=%.2f rl=%.2f",
                blended,
                symbol,
                ps.trend_score,
                ps.vol_score,
                size_mult,
                ps.regime,
                ps.stress_multiplier,
                rl_mult,
            )
            if not ps.last_price:
                logger.warning("ExecutionAgent: no last price for %s; skipping target sizing.", symbol)
                continue

            best_venue = self.venue_router.choose_best_venue(
                symbol=symbol,
                side=blended,
                notional_aud=notional_aud,
                state_meta=self.ctx.state.meta,
            )
            if best_venue:
                logger.info(
                    "ExecutionAgent: routing %s %s via %s (score=%.3f)",
                    blended, symbol, best_venue.venue.value, best_venue.score,
                )
                ps.meta["execution_venue"] = best_venue.venue.value
                ps.meta["execution_venue_score"] = best_venue.score
                ps.meta["execution_venue_reason"] = best_venue.reason
            else:
                ps.meta["execution_venue"] = "kraken"

            signed_notional = notional_aud if blended == "BUY" else -notional_aud
            target_size = signed_notional / ps.last_price
            target_positions[symbol] = (target_size, blended)

            await send_telegram_message(
                f"[TARGET] {symbol}: {target_size:.6f} (side={blended}, trend={ps.trend_score:.2f},"
                f" vol={ps.vol_score:.2f}, x{size_mult:.2f}, regime={ps.regime}, rl={rl_mult:.2f})"
            )

            reward = float(self.ctx.state.meta.get("last_trade_pnl", 0.0))

            # Record confirmed outcome for loss clustering & governance
            self.ctx.state.record_trade_outcome(pnl=reward)

            await self._update_rl_reward(reward)

        if target_positions:
            mode = os.getenv("BOT_MODE", "paper")
            logger.info("ExecutionAgent: setting targets %s mode=%s", target_positions, mode)

            if mode != "live":
                logger.info("ExecutionAgent: paper mode → no real order placed.")
                return

            account = next(iter(self.ctx.state.accounts.values()), None)

            for symbol, (target_size, blended) in target_positions.items():
                ps = self.ctx.state.pairs.get(symbol)
                if not ps or not ps.last_price:
                    logger.warning("ExecutionAgent: missing price for %s; cannot size order.", symbol)
                    continue

                current_size = 0.0
                if account:
                    current_position = account.positions.get(symbol)
                    if current_position:
                        current_size = current_position.size

                delta = target_size - current_size
                if abs(delta) < 1e-9:
                    logger.info("ExecutionAgent: %s already at target %.6f", symbol, target_size)
                    continue

                side = "BUY" if delta > 0 else "SELL"
                notional_aud = abs(delta) * ps.last_price

                venue_choice = self.venue_router.choose_best_venue(
                    symbol=symbol,
                    side=side,
                    notional_aud=notional_aud,
                    state_meta=self.ctx.state.meta,
                )
                venue = venue_choice.venue if venue_choice else Venue.KRAKEN

                # Create intent dict for metrics
                intent_dict = {
                    "symbol": symbol,
                    "side": blended,
                    "meta": {"strategy": "execution_agent"}
                }

                # Track order latency
                timer = order_latency_timer()
                res = await self.multi_executor.execute(
                    venue=venue,
                    symbol=symbol,
                    side=blended,
                    notional_quote=notional_aud,
                    futures=(venue == Venue.BINANCE),
                )
                timer.observe()

                if not res["ok"]:
                    logger.warning(
                        "ExecutionAgent: execution failed on %s → %s",
                        venue.value,
                        res.get("message", ""),
                    )
                    on_order_error(intent_dict)
                else:
                    on_order_submitted(intent_dict)

    def _blend_signals(self, ps: PairSnapshot, strat_weights: dict[str, float], mode: str) -> str | None:
        ml_signal = None
        if ps.meta:
            ml_signal = (ps.meta.get("ml_signal") or {}).get("side")

        signals = {
            "rsi": ps.last_signal_short,
            "trend": ps.last_signal_mid or ps.last_signal_long,
            "ml": ml_signal,
        }

        if not any(signals.values()):
            return None

        score = 0.0
        for key, sig in signals.items():
            weight = strat_weights.get(key, 0.0)
            if sig == "BUY":
                score += weight
            elif sig == "SELL":
                score -= weight

        threshold = 0.1
        if mode == "SAFE":
            threshold = 0.2
        elif mode == "AGGRESSIVE":
            threshold = 0.05

        if score > threshold:
            return "BUY"
        if score < -threshold:
            return "SELL"
        return "HOLD"

    def _volatility_size_multiplier(self, ps: PairSnapshot) -> float:
        t = ps.trend_score
        v = ps.vol_score
        if v > 0.8 and t < 0.2:
            return 0.3  # high vol, no trend → small size
        if t > 0.7 and 0.3 < v < 0.9:
            return 1.5  # good trend, moderate vol → bigger
        if t < 0.1:
            return 0.5  # no clear trend → half size
        return 1.0

    def _regime_sizing(self, ps: PairSnapshot) -> float:
        if ps.regime == RegimeLabel.HIGH_VOL:
            return 0.4
        if ps.regime == RegimeLabel.MEAN_REVERTING:
            return 0.7
        if ps.regime == RegimeLabel.TRENDING:
            return 1.2
        return 1.0

    async def _compute_notional_for_symbol(
        self, ps: PairSnapshot, base_notional_aud: float, size_mult: float
    ) -> float:
        win_rate = float(self.ctx.state.meta.get("rolling_win_rate", 0.5))
        drawdown_pct = 0.0
        if self.ctx.state.accounts:
            acct = next(iter(self.ctx.state.accounts.values()))
            drawdown_pct = float(acct.meta.get("drawdown_pct", acct.max_drawdown_pct))
        regime_score = float(ps.meta.get("regime_score", ps.trend_score))

        rl_mult = self.rlsizer.choose_multiplier(win_rate, drawdown_pct, regime_score)
        return base_notional_aud * size_mult * rl_mult

    async def _update_rl_reward(self, reward: float) -> None:
        win_rate_next = float(self.ctx.state.meta.get("rolling_win_rate", 0.5))
        drawdown_next = 0.0
        if self.ctx.state.accounts:
            acct = next(iter(self.ctx.state.accounts.values()))
            drawdown_next = float(acct.meta.get("drawdown_pct", acct.max_drawdown_pct))
        regime_next = 0.0
        # fall back to average of available regime scores
        if self.ctx.state.pairs:
            regime_next = float(
                sum(p.meta.get("regime_score", p.trend_score) for p in self.ctx.state.pairs.values())
                / max(len(self.ctx.state.pairs), 1)
            )

        self.rlsizer.update(reward, win_rate_next, drawdown_next, regime_next)
