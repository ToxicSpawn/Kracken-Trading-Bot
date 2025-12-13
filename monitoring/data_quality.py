from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime, timezone

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

@dataclass
class DataQualityStatus:
    ok: bool = True
    reason: str = "OK"
    last_update: str = field(default_factory=lambda: utcnow().isoformat())
    meta: Dict[str, Any] = field(default_factory=dict)

class DataQualityMonitor:
    def __init__(self, **kwargs: Any) -> None:
        # Accept but ignore legacy parameters for backward compatibility
        self._by_symbol: Dict[str, DataQualityStatus] = {}

    def update(self, symbol: str, **meta: Any) -> None:
        st = self._by_symbol.get(symbol) or DataQualityStatus()
        st.ok = True
        st.reason = "OK"
        st.last_update = utcnow().isoformat()
        if meta: st.meta.update(meta)
        self._by_symbol[symbol] = st

    def mark_bad(self, symbol: str, reason: str, **meta: Any) -> None:
        st = self._by_symbol.get(symbol) or DataQualityStatus()
        st.ok = False
        st.reason = reason or "BAD_DATA"
        st.last_update = utcnow().isoformat()
        if meta: st.meta.update(meta)
        self._by_symbol[symbol] = st

    def is_ok(self, symbol: str) -> bool:
        return (self._by_symbol.get(symbol) or DataQualityStatus()).ok

    def status(self, symbol: str) -> DataQualityStatus:
        """Get the current status for a symbol."""
        return self._by_symbol.get(symbol) or DataQualityStatus()

    def evaluate(self, symbol: str, timeframe: str, ohlcv: Any) -> Optional[Dict[str, Any]]:
        """Backward compatibility method for existing code.
        
        This method provides a compatibility layer for code that uses the old API.
        For new code, use update() and status() methods instead.
        """
        if not ohlcv or len(ohlcv) < 2:
            return None
        
        # Update status when data is available
        self.update(symbol, timeframe=timeframe, ohlcv_count=len(ohlcv))
        
        # Return minimal status dict for backward compatibility
        st = self.status(symbol)
        return {
            "last_ts": 0,
            "gap_ms": 0,
            "gap": False,
            "stale": not st.ok,
            "alert": not st.ok,
        }
