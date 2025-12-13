from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from agents.base import BaseAgent
from notifications.telegram import send_telegram_message

logger = logging.getLogger(__name__)


class LossClusterSupervisor(BaseAgent):
    """Supervisor that enforces PAUSE when loss clusters trigger.

    It watches state.meta for confirmed trade outcomes:
      - last_trade_pnl: float
      - last_trade_ts: ISO8601 string (optional but recommended)

    If last_trade_ts is present, it will de-duplicate outcomes by timestamp.
    Otherwise, it will only record when last_trade_pnl changes.

    Actions:
      - When cluster says pause: sets state.trading_enabled = False and sets state.meta['system_mode'] = 'PAUSED'
      - When pause expires: re-enables trading unless user has disabled trading_enabled manually.
    """

    def __init__(self, name, ctx, interval: float = 2.0) -> None:
        super().__init__(name, ctx, interval=interval)
        self._last_seen_ts: Optional[str] = None
        self._last_seen_pnl: Optional[float] = None
        self._paused_by_cluster: bool = False

    async def step(self) -> None:
        st = self.ctx.state
        meta = st.meta or {}

        # Ensure tracker exists
        lc = st.meta.get("loss_cluster")
        if lc is None:
            try:
                from core.loss_cluster import LossClusterTracker
                lc = LossClusterTracker()
                st.meta["loss_cluster"] = lc
            except Exception:
                return

        pnl = meta.get("last_trade_pnl")
        ts_iso = meta.get("last_trade_ts")

        if pnl is not None:
            # Dedup
            if ts_iso:
                if ts_iso != self._last_seen_ts:
                    self._last_seen_ts = ts_iso
                    lc.record_outcome(float(pnl))
            else:
                if self._last_seen_pnl is None or float(pnl) != float(self._last_seen_pnl):
                    self._last_seen_pnl = float(pnl)
                    lc.record_outcome(float(pnl))

        # Enforce pause if needed
        if lc.should_pause():
            if not self._paused_by_cluster:
                self._paused_by_cluster = True
                st.trading_enabled = False
                st.meta["system_mode"] = "PAUSED"
                st.meta["pause_reason"] = "LOSS_CLUSTER"
                st.meta["pause_until"] = lc.pause_until().isoformat() if lc.pause_until() else None
                logger.warning("LossClusterSupervisor: PAUSED due to loss cluster until %s", st.meta.get("pause_until"))
                await send_telegram_message(f"[PAUSE] Loss cluster triggered. Pausing until {st.meta.get('pause_until')}")
            return

        # Unpause if we paused and cooldown is over
        if self._paused_by_cluster:
            self._paused_by_cluster = False
            # Only re-enable if user didn't manually disable trading.
            st.trading_enabled = True
            st.meta["system_mode"] = "NORMAL"
            st.meta["pause_reason"] = None
            st.meta["pause_until"] = None
            logger.warning("LossClusterSupervisor: UNPAUSED (cluster cooldown ended)")
            await send_telegram_message("[PAUSE] Loss cluster cooldown ended. Resuming trading.")

