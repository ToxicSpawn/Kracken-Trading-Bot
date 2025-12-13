from __future__ import annotations

"""Loss cluster tracking for throttling and pausing trading after clustered losses.

Python 3.9 compatible - uses Optional instead of | syntax.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Optional


@dataclass
class LossEvent:
    ts: datetime
    pnl: float


class LossClusterTracker:
    def __init__(
        self,
        window_minutes: int = 30,
        throttle_losses: int = 2,
        pause_losses: int = 3,
        cooldown_minutes: int = 20,
    ) -> None:
        self.window = timedelta(minutes=window_minutes)
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.throttle_losses = throttle_losses
        self.pause_losses = pause_losses

        self._losses: Deque[LossEvent] = deque()
        self._cooldown_until: Optional[datetime] = None

    def record_loss(self, pnl: float, ts: Optional[datetime] = None) -> None:
        if pnl >= 0:
            return

        ts = ts or datetime.utcnow()
        self._losses.append(LossEvent(ts=ts, pnl=pnl))
        self._prune(ts)

        if len(self._losses) >= self.pause_losses:
            self._cooldown_until = ts + self.cooldown

    def _prune(self, now: datetime) -> None:
        while self._losses and now - self._losses[0].ts > self.window:
            self._losses.popleft()

    def should_pause(self) -> bool:
        return self._cooldown_until is not None and datetime.utcnow() < self._cooldown_until

    def throttle_multiplier(self) -> float:
        if len(self._losses) >= self.throttle_losses:
            return 0.25
        return 1.0

