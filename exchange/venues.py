from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Venue(str, Enum):
    KRAKEN = "kraken"
    BINANCE = "binance"
    COINBASE = "coinbase"


@dataclass
class VenueConfig:
    venue: Venue
    enabled: bool = True
    # rough latency hints (you can measure and update)
    base_latency_ms: float = 50.0
    fee_bps: float = 8.0  # trading fee in basis points (0.08% etc.)
    max_notional_per_order: float = 5_000.0


@dataclass
class Instrument:
    """
    Logical instrument, e.g. 'SOL/AUD' or 'BTC/USD', with mapping to per-venue
    symbols (like 'SOLAUD', 'SOLUSDT', 'SOL-USD', etc.).
    """

    symbol: str
    base: str
    quote: str
    venues: Dict[Venue, str]  # venue -> venue_specific_symbol


# Example: you can load these from config later
DEFAULT_VENUES: Dict[Venue, VenueConfig] = {
    Venue.KRAKEN: VenueConfig(venue=Venue.KRAKEN, base_latency_ms=60.0, fee_bps=16.0),
    Venue.BINANCE: VenueConfig(venue=Venue.BINANCE, base_latency_ms=40.0, fee_bps=10.0),
    Venue.COINBASE: VenueConfig(venue=Venue.COINBASE, base_latency_ms=50.0, fee_bps=15.0),
}
