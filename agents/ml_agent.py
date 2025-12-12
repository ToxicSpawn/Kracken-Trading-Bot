from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import List

from agents.base import BaseAgent
from core.state import PairSnapshot
from exchange.data_feed import DataFeed
from strategies.xgb_ml_strategy import MlConfig, XgbMlStrategy

logger = logging.getLogger(__name__)


class MlAgent(BaseAgent):
    """
    MLAgent — v120000 Omega Edition

    Upgrades:
    - Reads Omega mode (OFF/SAFE/AGGRESSIVE/SHOCK)
    - Only runs ML on Omega-allowed pairs
    - Skips ML updates in dangerous markets (chaotic volatility, data-quality issues)
    - Writes ML signal in Omega-consistent format for ExecutionAgent blending
    - Dynamic SAFE mode throttling
    """

    def __init__(self, name: str, ctx, symbols: List[str], interval: float = 60.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self.symbols = symbols
        self.data_feed = DataFeed()
        self.model_path = self._resolve_model_path()
        self._strategies: dict[str, XgbMlStrategy] = {
            sym: XgbMlStrategy(sym, MlConfig(model_path=self.model_path))
            for sym in symbols
        }
        self._safe_skip_flag = False

    def _resolve_model_path(self) -> Path:
        meta_path = self.ctx.state.meta.get("ml_model_path")
        env_path = os.getenv("ML_MODEL_PATH")
        if meta_path:
            return Path(meta_path)
        if env_path:
            return Path(env_path)
        return Path("models") / "xgb_alpha.pkl"

    async def step(self) -> None:  # type: override
        router = self.ctx.state.meta.get("router", {})
        omega = self.ctx.state.meta.get("omega", {})

        mode = omega.get("mode") or router.get("mode", "NORMAL")
        allowed_pairs = set(
            omega.get("allowed_pairs") or router.get("enabled_pairs") or self.symbols
        )

        dq_flags = int(self.ctx.state.meta.get("data_quality_flags", 0))
        regime_l2 = (self.ctx.state.meta.get("regime_l2") or {})
        vol_state = str(regime_l2.get("vol_state", "normal"))
        shock = bool(regime_l2.get("shock", False))

        # Mode OFF → skip all ML activity
        if mode == "OFF":
            logger.info("MLAgent: Omega mode OFF → skipping ML cycle")
            return

        # Skip during extreme market turbulence
        if shock or vol_state == "chaotic" or dq_flags > 10:
            logger.warning(
                "MLAgent: Skipped ML update (shock=%s, vol=%s, dq_flags=%s)",
                shock, vol_state, dq_flags
            )
            return

        # SAFE mode: throttle ML to run every *other* cycle
        if mode == "SAFE":
            if self._safe_skip_flag:
                self._safe_skip_flag = False
                return
            self._safe_skip_flag = True
        else:
            self._safe_skip_flag = False

        # ---- Run ML per symbol ----
        for sym, strat in self._strategies.items():
            if sym not in allowed_pairs:
                logger.debug("MLAgent: %s blocked by Omega/router", sym)
                continue

            ohlcv = await self.data_feed.get_recent_ohlcv(sym, "5m", limit=200)
            if not ohlcv:
                continue

            signal = strat.generate_signal(ohlcv)
            ps = self.ctx.state.pairs.setdefault(sym, PairSnapshot(symbol=sym))

            meta = ps.meta or {}
            meta["ml_signal"] = {
                "side": signal.signal.name,
                "confidence": signal.confidence,
            }
            ps.meta = meta

            logger.info(
                "MLAgent: %s ML %s conf=%.3f (Ωmode=%s)",
                sym, signal.signal.name, signal.confidence, mode
            )

    async def run_loop(self) -> None:
        logger.info("MLAgent %s loop started", self.name)
        while self._running:
            try:
                await self.step()
            except Exception as exc:
                logger.exception("MLAgent error: %s", exc)
            await asyncio.sleep(self.interval)
