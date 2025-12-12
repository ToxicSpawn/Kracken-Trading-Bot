from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from exchange.venues import Venue


@dataclass
class SpreadSnapshot:
    base_symbol: str          # e.g. "SOL/AUD" (bot symbol)
    leg_buy_venue: Venue      # where we'd buy
    leg_sell_venue: Venue     # where we'd sell
    buy_price: float
    sell_price: float
    gross_spread_bps: float   # (sell - buy) / mid * 10_000
    net_spread_bps: float     # after fees estimate
    mid: float


class CrossVenueArbEngine:
    """
    v2,000,000 Cross-Venue Arbitrage Engine

    Reads:
      state.meta["venue_book"][symbol][venue] = {
        best_bid, best_ask, spread_bps, mid_price, ...
      }

    Produces:
      Best spread opportunity (if any) for a base_symbol.
    """

    def __init__(self, fee_bps: Dict[Venue, float]) -> None:
        self.fee_bps = fee_bps

    def find_best_spread(self, symbol: str, venue_book: dict) -> Optional[SpreadSnapshot]:
        venues_for_sym: Dict[str, dict] = venue_book.get(symbol, {})
        if len(venues_for_sym) < 2:
            return None

        best: Optional[SpreadSnapshot] = None

        # Brute-force pairwise venue comparison
        for v_buy_name, b in venues_for_sym.items():
            for v_sell_name, a in venues_for_sym.items():
                if v_buy_name == v_sell_name:
                    continue

                buy_venue = Venue(v_buy_name)
                sell_venue = Venue(v_sell_name)

                buy_px = float(b["best_ask"])
                sell_px = float(a["best_bid"])
                if buy_px <= 0 or sell_px <= 0:
                    continue

                mid = (buy_px + sell_px) / 2.0
                gross_spread_bps = (sell_px - buy_px) / mid * 10_000.0

                fee_buy = self.fee_bps.get(buy_venue, 10.0)
                fee_sell = self.fee_bps.get(sell_venue, 10.0)
                approx_cost_bps = fee_buy + fee_sell + 3.0  # +3bps slippage buffer

                net_spread_bps = gross_spread_bps - approx_cost_bps
                if net_spread_bps <= 0:
                    continue

                snap = SpreadSnapshot(
                    base_symbol=symbol,
                    leg_buy_venue=buy_venue,
                    leg_sell_venue=sell_venue,
                    buy_price=buy_px,
                    sell_price=sell_px,
                    gross_spread_bps=gross_spread_bps,
                    net_spread_bps=net_spread_bps,
                    mid=mid,
                )

                if best is None or snap.net_spread_bps > best.net_spread_bps:
                    best = snap

        return best
