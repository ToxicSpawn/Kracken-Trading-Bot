from __future__ import annotations
from exchange.venues import Venue


class VenueSymbolMapper:
    """
    Auto-map a bot symbol (like 'SOL/AUD') to each venue’s expected symbol.
    v1,000,000+ edition.

    • Kraken expects: SOL/AUD
    • Binance spot:  SOL/USDT
    • Binance futures: SOL/USDT
    • Coinbase: SOL/USD

    You can extend this later with FX, indices, futures, synthetics, whatever.
    """

    QUOTE_MAP = {
        # Crypto spot/futures
        Venue.KRAKEN: {"default": "AUD", "overrides": {}},
        Venue.BINANCE: {"default": "USDT", "overrides": {}},
        Venue.COINBASE: {"default": "USD", "overrides": {}},
        # FX/indices scaffolding (add overrides per base later)
    }

    @staticmethod
    def map_symbol(base_symbol: str, venue: Venue, futures: bool = False) -> str:
        """
        base_symbol is the bot symbol (ex: 'SOL/AUD').
        We remap only the quote part.
        """
        try:
            base, _ = base_symbol.split("/")
        except Exception:
            raise ValueError(f"Invalid bot symbol: {base_symbol}")

        venue_quotes = VenueSymbolMapper.QUOTE_MAP.get(venue)
        if not venue_quotes:
            raise ValueError(f"Unknown venue: {venue}")

        quote = venue_quotes["overrides"].get(base, venue_quotes.get("default"))
        if not quote:
            raise ValueError(f"No quote mapping for {base} on {venue}")

        return f"{base}/{quote}"
