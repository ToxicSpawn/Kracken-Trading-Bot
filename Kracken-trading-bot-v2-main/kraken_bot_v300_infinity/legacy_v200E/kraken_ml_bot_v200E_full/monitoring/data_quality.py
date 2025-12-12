from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List


def _timeframe_ms(timeframe: str) -> int:
    unit = timeframe[-1]
    try:
        value = int(timeframe[:-1])
    except ValueError:
        return 300_000

    if unit == "s":
        return value * 1000
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 3_600_000
    if unit == "d":
        return value * 86_400_000
    return 300_000


@dataclass
class DataQualityStatus:
    last_ts: int
    gap_ms: int
    gap_detected: bool
    stale: bool
    alert: bool = False

    def as_dict(self) -> Dict[str, int | bool]:
        return {
            "last_ts": self.last_ts,
            "gap_ms": self.gap_ms,
            "gap": self.gap_detected,
            "stale": self.stale,
            "alert": self.alert,
        }


class DataQualityMonitor:
    def __init__(
        self,
        *,
        enabled: bool = True,
        alert_interval_sec: float = 1800.0,
        gap_multiplier: float = 1.4,
        stale_multiplier: float = 2.0,
    ) -> None:
        self.enabled = enabled
        self.alert_interval_sec = alert_interval_sec
        self.gap_multiplier = gap_multiplier
        self.stale_multiplier = stale_multiplier
        self._last_alert: Dict[str, float] = {}

    def _should_alert(self, key: str) -> bool:
        now = time.time()
        last = self._last_alert.get(key, 0)
        if now - last >= self.alert_interval_sec:
            self._last_alert[key] = now
            return True
        return False

    def evaluate(self, symbol: str, timeframe: str, ohlcv: List[list]) -> Dict[str, int | bool] | None:
        if not self.enabled or len(ohlcv) < 2:
            return None

        expected_ms = _timeframe_ms(timeframe)
        last_ts = int(ohlcv[-1][0])
        prev_ts = int(ohlcv[-2][0])
        gap_ms = last_ts - prev_ts

        gap_detected = gap_ms > expected_ms * self.gap_multiplier
        now_ms = int(time.time() * 1000)
        stale = (now_ms - last_ts) > expected_ms * self.stale_multiplier

        alert = False
        if gap_detected or stale:
            key = f"{symbol}:{timeframe}"
            alert = self._should_alert(key)

        status = DataQualityStatus(
            last_ts=last_ts,
            gap_ms=gap_ms,
            gap_detected=gap_detected,
            stale=stale,
            alert=alert,
        )
        return status.as_dict()
