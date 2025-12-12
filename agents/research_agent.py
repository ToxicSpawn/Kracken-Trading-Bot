from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any

from agents.base import BaseAgent
from analytics.research.auto_research import AutoResearchEngine, ResearchConfig

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    v900000 ResearchAgent

    - Runs occasional research passes (not constantly, to avoid CPU spikes).
    - Uses AutoResearchEngine to grid-search strategy params.
    - Writes best params into state.meta["research"]["best_params"] so Omega can see them.
    - You can configure it to only run at certain UTC hours if you want.
    """

    def __init__(
        self,
        name: str,
        ctx,
        interval: float = 3600.0,  # run at most once per hour
        enabled: bool = True,
    ) -> None:
        super().__init__(name, ctx, interval=interval)
        self.enabled = enabled
        self.engine = AutoResearchEngine(ResearchConfig())
        self._running_job = False

    async def step(self) -> None:  # type: override
        if not self.enabled:
            return

        # Don’t start a new job if one is still running.
        if self._running_job:
            return

        # Simple throttle: only run if bot is not in deep drawdown
        dd_pct = float(self.ctx.state.meta.get("drawdown_pct", 0.0))
        if dd_pct <= -35.0:
            logger.info("ResearchAgent: skipping research (deep drawdown %.1f%%)", dd_pct)
            return

        # You can also check Omega mode; maybe only run in NORMAL/SAFE:
        omega = self.ctx.state.meta.get("omega", {})
        mode = omega.get("mode", "NORMAL")
        if mode not in ("NORMAL", "SAFE"):
            logger.info("ResearchAgent: skipping (Ωmode=%s)", mode)
            return

        # Kick off background job
        self._running_job = True
        asyncio.create_task(self._run_research_job())

    async def _run_research_job(self) -> None:
        try:
            logger.info("ResearchAgent: starting research job")

            # Example parameter grid for RSI/Trend strategy.
            # Adapt names to your backtester’s expected params.
            param_grid: Dict[str, List[Any]] = {
                "rsi_period": [8, 10, 14],
                "rsi_buy": [25, 30],
                "rsi_sell": [70, 75],
                "trend_ma_fast": [10, 20],
                "trend_ma_slow": [50, 100],
            }

            best = await asyncio.to_thread(self.engine.run_grid_search, param_grid)

            logger.info("ResearchAgent: best params=%s score=%.3f", best.params, best.score)

            research_meta = self.ctx.state.meta.setdefault("research", {})
            research_meta["best_params"] = best.params
            research_meta["best_score"] = best.score
            # Optionally track last run time:
            research_meta["last_run_ts"] = int(self.ctx.loop.time())

        except Exception as exc:  # noqa: BLE001
            logger.exception("ResearchAgent job error: %s", exc)
        finally:
            self._running_job = False
