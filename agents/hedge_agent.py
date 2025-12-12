from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from agents.base import BaseAgent
from analytics.portfolio.correlation import CorrelationConfig, CorrelationEngine
from analytics.portfolio.optimizer import PortfolioOptimConfig, PortfolioOptimizer
from exchange.data_feed import DataFeed
from exchange.order_executor import OrderExecutor

logger = logging.getLogger(__name__)


class HedgeAgent(BaseAgent):
    """
    v12000 Hedge / Overlay agent.

    - Looks at account exposures per symbol.
    - Builds correlation matrix from recent prices.
    - If exposures are highly correlated, proposes small hedging trades
      to reduce concentration risk.
    """

    def __init__(self, name: str, ctx, symbols: List[str], interval: float = 300.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self.symbols = symbols
        self.data_feed = DataFeed()
        self.corr_engine = CorrelationEngine(CorrelationConfig())
        self.optimizer = PortfolioOptimizer(PortfolioOptimConfig())
        # smaller base notional for hedges – we don't want huge hedges
        self.executor = OrderExecutor()

    async def step(self) -> None:  # type: ignore[override]
        state = self.ctx.state
        if not state.accounts:
            return

        router = state.meta.get("router", {})
        mode = router.get("mode", "NORMAL")
        enabled_pairs = set(router.get("enabled_pairs") or self.symbols)
        risk_mult = float(router.get("risk_multiplier", 1.0))

        if mode == "OFF":
            logger.info("HedgeAgent: router mode OFF; skipping hedges.")
            return

        acct = next(iter(state.accounts.values()))

        exposures: Dict[str, float] = {}
        for sym, pos in (acct.positions or {}).items():
            if pos.entry_price is None or pos.size == 0:
                continue
            exposures[sym] = exposures.get(sym, 0.0) + pos.entry_price * pos.size

        if enabled_pairs:
            exposures = {sym: exp for sym, exp in exposures.items() if sym in enabled_pairs}

        if not exposures:
            return

        price_series: Dict[str, pd.Series] = {}
        for sym in exposures.keys():
            ohlcv = await self.data_feed.get_recent_ohlcv(sym, "5m", limit=300)
            if not ohlcv:
                continue
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            price_series[sym] = df["close"].astype(float)

        corr_matrix = self.corr_engine.compute_matrix(price_series)
        if corr_matrix.empty:
            return

        proposals = self.corr_engine.propose_hedges(exposures, corr_matrix)
        if not proposals:
            return

        logger.info("HedgeAgent: hedge proposals=%s", proposals)

        for source_sym, hedge_sym, frac in proposals:
            src_exp = exposures.get(source_sym, 0.0)
            if abs(src_exp) < 1e-6:
                continue
            if hedge_sym not in enabled_pairs:
                logger.debug("HedgeAgent: %s disabled by router; skipping hedge.", hedge_sym)
                continue

            direction = "SELL" if src_exp > 0 else "BUY"  # simple opposite
            hedge_notional = abs(src_exp) * frac * 0.1 * risk_mult  # only hedge a small fraction
            if mode == "SAFE":
                hedge_notional *= 0.6  # extra conservative in SAFE mode

            logger.info(
                "HedgeAgent: hedging %s exposure %.2f via %s %s notional=%.2f",
                source_sym,
                src_exp,
                hedge_sym,
                direction,
                hedge_notional,
            )

            await self.executor.execute(hedge_sym, direction, notional_aud=hedge_notional)
