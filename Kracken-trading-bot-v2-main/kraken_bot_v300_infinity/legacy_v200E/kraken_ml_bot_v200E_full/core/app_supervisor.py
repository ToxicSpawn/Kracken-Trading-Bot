from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, List, Optional

from core.utils import utcnow


@dataclass
class HealthRule:
    agent: str
    max_staleness_seconds: float
    severity: str = "CRITICAL"  # CRITICAL -> PAUSED, WARN -> DEGRADED


async def monitor_health(state, rules: List[HealthRule], interval_seconds: float = 5.0):
    while True:
        now = utcnow()
        worst = None
        for r in rules:
            h = state.agent_health.get(r.agent, {})
            ts = h.get("last_heartbeat")
            if not ts:
                worst = (r, "missing")
                break
            try:
                hb = utcnow().__class__.fromisoformat(ts)
            except Exception:
                worst = (r, "bad_timestamp")
                break
            age = (now - hb).total_seconds()
            if age > r.max_staleness_seconds:
                worst = (r, f"stale:{age:.1f}s")
                break

        if worst:
            rule, why = worst
            if rule.severity == "CRITICAL":
                state.set_system_mode("PAUSED", reason=f"HEALTH_{rule.agent}_{why}")
            else:
                state.set_system_mode("DEGRADED", reason=f"HEALTH_{rule.agent}_{why}")
        await asyncio.sleep(interval_seconds)

