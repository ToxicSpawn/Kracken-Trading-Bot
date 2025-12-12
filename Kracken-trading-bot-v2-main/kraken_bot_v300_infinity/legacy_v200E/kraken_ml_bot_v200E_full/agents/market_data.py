from __future__ import annotations

import logging
import os
from typing import Any, Dict

import yaml

from agents.base import BaseAgent
from analytics import MicrostructureProfiler, RegimeClassifier, StressTester
from core.state import PairSnapshot
from exchange.data_feed import DataFeed
from monitoring.data_quality import DataQualityMonitor
from notifications.telegram import send_telegram_message

logger = logging.getLogger(__name__)


class MarketDataAgent(BaseAgent):
    def __init__(self, name, ctx, symbols, interval: float = 5.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self.symbols = symbols
        self.feed = DataFeed()
        self.regime = RegimeClassifier()
        self.micro = MicrostructureProfiler()
        self.stress = StressTester()
        monitor_cfg = self._load_monitoring_config()
        self.monitor = DataQualityMonitor(
            enabled=monitor_cfg.get("enabled", True),
            alert_interval_sec=float(monitor_cfg.get("alert_interval_sec", 1800)),
            gap_multiplier=float(monitor_cfg.get("gap_multiplier", 1.4)),
            stale_multiplier=float(monitor_cfg.get("stale_multiplier", 2.0)),
        )

    @staticmethod
    def _load_monitoring_config() -> Dict[str, Any]:
        path = "config/monitoring.yaml"
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        monitoring_cfg = cfg.get("monitoring", {})
        return monitoring_cfg.get("data_quality", cfg.get("data_quality", {}))

    async def step(self) -> None:
        for symbol in self.symbols:
            ps = self.ctx.state.pairs.setdefault(symbol, PairSnapshot(symbol=symbol))
            ohlcv = await self.feed.get_recent_ohlcv(symbol, timeframe="5m", limit=180)
            if not ohlcv:
                continue
            last_close = float(ohlcv[-1][4])
            ps.last_price = last_close
            logger.debug("MarketData: %s last_price=%s", symbol, last_close)

            # Regime + microstructure overlays to feed downstream sizing/guardrails.
            try:
                ps.regime = self.regime.classify(ohlcv)
                micro = self.micro.evaluate(ohlcv)
                ps.liquidity_score = micro.liquidity_score
                ps.slippage_bps = micro.slippage_bps
                stress = self.stress.evaluate(micro.volatility, micro.slippage_bps)
                ps.stress_multiplier = stress.size_multiplier
                ps.meta.update(
                    {
                        "microstructure": {
                            "volatility": micro.volatility,
                            "spread_proxy_bps": micro.spread_proxy_bps,
                            "slippage_bps": micro.slippage_bps,
                            "liquidity_score": micro.liquidity_score,
                        },
                        "regime": ps.regime,
                        "stress_reason": stress.reason,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("MarketData: overlay error for %s: %s", symbol, exc)

            status = self.monitor.evaluate(symbol, "5m", ohlcv)
            if status:
                dq_state = self.ctx.state.meta.setdefault("data_quality", {})
                sym_state = dq_state.setdefault(symbol, {})
                sym_state["5m"] = status

                if status.get("alert"):
                    gap_sec = status.get("gap_ms", 0) / 1000
                    stale = status.get("stale", False)
                    msg = (
                        f"[DataQuality] Gap detected for {symbol} 5m feed. "
                        f"Gap={gap_sec:.1f}s stale={stale}"
                    )
                    logger.warning(msg)
                    await send_telegram_message(msg)
