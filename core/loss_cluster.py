from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Optional


@dataclass
class LossEvent:
    ts: datetime
    pnl: float


class LossClusterTracker:
    """Detect clusters of *realized* losses and recommend throttle/pause.

    Conservative defaults (research-safe):
      - Throttle after 2 losses within 30 minutes
      - Pause after 3 losses within 30 minutes
      - Pause duration 60 minutes (cooldown)

    Notes:
    - This tracker is intentionally exchange/strategy-agnostic.
    - You can feed it from any confirmed trade outcome source.
    """

    def __init__(
        self,
        window_minutes: int = 30,
        throttle_losses: int = 2,
        pause_losses: int = 3,
        pause_minutes: int = 60,
        throttle_multiplier: float = 0.25,
    ) -> None:
        self.window = timedelta(minutes=int(window_minutes))
        self.throttle_losses = int(throttle_losses)
        self.pause_losses = int(pause_losses)
        self.pause_duration = timedelta(minutes=int(pause_minutes))
        self._throttle_mult = float(throttle_multiplier)

        self._losses: Deque[LossEvent] = deque()
        self._pause_until: Optional[datetime] = None

    def record_outcome(self, pnl: float, ts: Optional[datetime] = None) -> None:
        """Record a confirmed trade outcome (pnl). Negative pnl counts as a loss."""
        if pnl is None:
            return
        ts = ts or datetime.utcnow()
        self._prune(ts)
        if float(pnl) < 0.0:
            self._losses.append(LossEvent(ts=ts, pnl=float(pnl)))
            self._prune(ts)
            if len(self._losses) >= self.pause_losses:
                self._pause_until = ts + self.pause_duration

    def _prune(self, now: datetime) -> None:
        while self._losses and (now - self._losses[0].ts) > self.window:
            self._losses.popleft()

    def should_pause(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.utcnow()
        return self._pause_until is not None and now < self._pause_until

    def pause_until(self) -> Optional[datetime]:
        return self._pause_until

    def throttle_multiplier(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.utcnow()
        self._prune(now)
        if len(self._losses) >= self.throttle_losses:
            return self._throttle_mult
        return 1.0

    def snapshot(self) -> dict:
        """Small serializable snapshot for metrics/debug."""
        return {
            "loss_count_window": len(self._losses),
            "pause_until": self._pause_until.isoformat() if self._pause_until else None,
            "throttle_mult": self._throttle_mult,
            "window_minutes": int(self.window.total_seconds() // 60),
            "throttle_losses": self.throttle_losses,
            "pause_losses": self.pause_losses,
            "pause_minutes": int(self.pause_duration.total_seconds() // 60),
        }
