from __future__ import annotations

from prometheus_client import start_http_server, Gauge

from analytics.scorecard import BotScorecard
from core.global_state import get_global_state

pnl_realized_gauge = Gauge("pnl_total_realized", "Total realized PnL (AUD)")
pnl_unrealized_gauge = Gauge("pnl_total_unrealized", "Total unrealized PnL (AUD)")
trades_count_gauge = Gauge("trades_count", "Number of trades executed (paper+live)")
data_gap_gauge = Gauge(
    "data_feed_gap",
    "Data gap detected for symbol/timeframe (1=yes)",
    labelnames=["symbol", "timeframe"],
)
data_stale_gauge = Gauge(
    "data_feed_stale",
    "Data staleness detected for symbol/timeframe (1=yes)",
    labelnames=["symbol", "timeframe"],
)
bot_health_score_gauge = Gauge(
    "bot_health_score",
    "Composite bot health score (0-20)",
)


def update_metrics(scorecard: BotScorecard | None = None) -> None:
    st = get_global_state()
    pnl_realized_gauge.set(st.total_realized_pnl)
    pnl_unrealized_gauge.set(st.total_unrealized_pnl)
    trades_count_gauge.set(st.meta.get("trades", 0))
    if scorecard:
        bot_health_score_gauge.set(scorecard.score)
    else:
        stored_score = st.meta.get("scorecard", {}).get("score")
        if stored_score is not None:
            bot_health_score_gauge.set(float(stored_score))

    dq = st.meta.get("data_quality", {})
    for symbol, frames in dq.items():
        for timeframe, status in frames.items():
            gap_val = 1 if status.get("gap") else 0
            stale_val = 1 if status.get("stale") else 0
            data_gap_gauge.labels(symbol=symbol, timeframe=timeframe).set(gap_val)
            data_stale_gauge.labels(symbol=symbol, timeframe=timeframe).set(stale_val)


def start_metrics_server(port: int) -> None:
    start_http_server(port)
