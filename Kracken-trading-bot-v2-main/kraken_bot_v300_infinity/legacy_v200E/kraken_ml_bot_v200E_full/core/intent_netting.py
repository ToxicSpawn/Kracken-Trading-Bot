from __future__ import annotations

"""Intent netting/compression for execution optimization.

Compatibility note:
- Uses Optional[X] instead of X | None for Python 3.9 compatibility.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NettingConfig:
    # Only net intents with same (account, symbol, type, venue)
    max_batch: int = 250
    # If true, keeps separate BUY/SELL legs (no cross-net); default nets to signed qty.
    keep_sides_separate: bool = False


def _signed_qty(side: str, qty: float) -> float:
    if side.upper() == "BUY":
        return float(qty)
    if side.upper() == "SELL":
        return -float(qty)
    raise ValueError(f"Unknown side: {side}")


def net_intents(intents: List[Dict[str, Any]], cfg: Optional[NettingConfig] = None) -> List[Dict[str, Any]]:
    cfg = cfg or NettingConfig()
    if not intents:
        return []

    buckets: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for it in intents[: cfg.max_batch]:
        account = it.get("account")
        symbol = it.get("symbol")
        typ = it.get("type", "market")
        venue = it.get("venue")
        side = (it.get("side") or "").upper()
        qty = float(it.get("qty", 0.0))
        if qty == 0.0 or not symbol or not side:
            continue

        key = (account, symbol, typ, venue, side) if cfg.keep_sides_separate else (account, symbol, typ, venue)

        if key not in buckets:
            # copy base intent
            base = dict(it)
            base["_net_signed_qty"] = _signed_qty(side, qty) if not cfg.keep_sides_separate else 0.0
            base["qty"] = float(qty)
            buckets[key] = base
        else:
            b = buckets[key]
            if cfg.keep_sides_separate:
                b["qty"] = float(b.get("qty", 0.0)) + qty
            else:
                b["_net_signed_qty"] = float(b.get("_net_signed_qty", 0.0)) + _signed_qty(side, qty)

            # merge exposure_delta and meta in a conservative way
            b["exposure_delta"] = float(b.get("exposure_delta", 0.0)) + float(it.get("exposure_delta", 0.0))
            b_meta = b.get("meta") or {}
            it_meta = it.get("meta") or {}
            if isinstance(b_meta, dict) and isinstance(it_meta, dict):
                # keep first, but record sources
                srcs = set(b_meta.get("sources", []))
                srcs.add(it.get("source"))
                b_meta["sources"] = sorted([s for s in srcs if s])
                b["meta"] = b_meta

    out: List[Dict[str, Any]] = []
    for b in buckets.values():
        if cfg.keep_sides_separate:
            out.append(b)
            continue

        signed = float(b.pop("_net_signed_qty", 0.0))
        if signed == 0.0:
            continue
        b["side"] = "BUY" if signed > 0 else "SELL"
        b["qty"] = abs(signed)
        out.append(b)

    return out

