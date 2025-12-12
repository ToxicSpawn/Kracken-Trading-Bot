from __future__ import annotations

import logging
from typing import Dict, List, Optional

from analytics.microstructure_venues import VenueBookSnapshot, VenueScore, score_venue
from exchange.venues import Venue, VenueConfig, DEFAULT_VENUES

logger = logging.getLogger(__name__)


class VenueRouter:
    """
    v1,000,000 Venue Router.

    - Reads per-venue microstructure snapshots from state.meta
      (populated by a multi-venue book agent).
    - Uses VenueConfig (fee, latency hints).
    - Returns best venue for a trade.

    Expects state.meta["venue_book"][symbol][venue_name] = { ... snapshot fields ... }
    """

    def __init__(self, venue_configs: Dict[Venue, VenueConfig] | None = None) -> None:
        self.venue_configs = venue_configs or DEFAULT_VENUES

    def choose_best_venue(
        self,
        symbol: str,
        side: str,
        notional_aud: float,
        state_meta: dict,
    ) -> Optional[VenueScore]:
        raw = (
            state_meta
            .get("venue_book", {})
            .get(symbol, {})
        )
        if not raw:
            logger.debug("VenueRouter: no venue_book for %s", symbol)
            return None

        scores: List[VenueScore] = []
        for venue_name, snap in raw.items():
            try:
                venue = Venue(venue_name)
            except ValueError:
                continue

            cfg = self.venue_configs.get(venue)
            if not cfg or not cfg.enabled:
                continue

            # Hard cap per order
            if notional_aud > cfg.max_notional_per_order:
                logger.debug(
                    "VenueRouter: skip %s (notional %.2f > max %.2f)",
                    venue_name, notional_aud, cfg.max_notional_per_order,
                )
                continue

            # Build snapshot object
            vs = VenueBookSnapshot(
                venue=venue,
                symbol=symbol,
                best_bid=float(snap["best_bid"]),
                best_ask=float(snap["best_ask"]),
                bid_size_quote=float(snap["bid_size_quote"]),
                ask_size_quote=float(snap["ask_size_quote"]),
                spread_bps=float(snap["spread_bps"]),
                mid_price=float(snap["mid_price"]),
                latency_ms=float(snap.get("latency_ms", cfg.base_latency_ms)),
            )
            scores.append(
                score_venue(
                    vs,
                    fee_bps=cfg.fee_bps,
                    extra_latency_penalty_ms=0.0,  # you can add path-specific latency here
                )
            )

        if not scores:
            return None

        best = max(scores, key=lambda s: s.score)
        logger.info(
            "VenueRouter: %s side=%s notional=%.2f → %s score=%.3f (%s)",
            symbol, side, notional_aud, best.venue.value, best.score, best.reason,
        )
        return best
