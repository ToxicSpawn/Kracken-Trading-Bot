from __future__ import annotations

from dataclasses import dataclass

from exchange.venues import Venue


@dataclass
class VenueBookSnapshot:
    venue: Venue
    symbol: str
    best_bid: float
    best_ask: float
    bid_size_quote: float
    ask_size_quote: float
    spread_bps: float
    mid_price: float
    latency_ms: float


@dataclass
class VenueScore:
    venue: Venue
    symbol: str
    score: float
    reason: str


def score_venue(snapshot: VenueBookSnapshot, fee_bps: float, extra_latency_penalty_ms: float) -> VenueScore:
    """
    Simple scoring:
      - narrower spreads better
      - deeper book better
      - lower latency better
      - lower fee better

    You can make this more sophisticated later (slippage models, impact, etc.).
    """
    spread = snapshot.spread_bps
    depth = min(snapshot.bid_size_quote, snapshot.ask_size_quote)
    latency = snapshot.latency_ms + extra_latency_penalty_ms

    # Normalize roughly
    depth_term = min(depth / 5_000.0, 1.0)
    spread_term = max(0.0, 1.0 - spread / 20.0)  # 0 at 20bps
    fee_term = max(0.0, 1.0 - fee_bps / 30.0)
    latency_term = max(0.0, 1.0 - latency / 200.0)

    score = 0.4 * spread_term + 0.3 * depth_term + 0.15 * fee_term + 0.15 * latency_term
    reason = (
        f"spread={spread:.2f}bps depth={depth:.0f} latency={latency:.1f}ms "
        f"fee={fee_bps:.1f}bps"
    )
    return VenueScore(venue=snapshot.venue, symbol=snapshot.symbol, score=score, reason=reason)
