from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from core.global_state import get_global_state


app = FastAPI(title="Kraken ML Bot v200E Dashboard")


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return '''
    <html>
      <head><title>v200E Dashboard</title></head>
      <body>
        <h1>Kraken ML Bot v200E</h1>
        <p>Multi-agent, trend/vol-aware dry-run bot.</p>
        <ul>
          <li><a href="/api/state">/api/state</a> – Global state JSON</li>
        </ul>
      </body>
    </html>
    '''


@app.get("/api/state")
async def get_state() -> dict:
    st = get_global_state()
    accounts = []
    for name, acc in st.accounts.items():
        accounts.append({
            "name": acc.name,
            "equity": acc.equity,
            "balance": acc.balance,
            "realized_pnl": acc.realized_pnl,
            "unrealized_pnl": acc.unrealized_pnl,
            "max_drawdown_pct": acc.max_drawdown_pct,
            "risk_multiplier": acc.risk_multiplier,
            "positions": {
                sym: {
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "realized_pnl": pos.realized_pnl,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "meta": pos.meta,
                }
                for sym, pos in acc.positions.items()
            },
        })

    pairs = []
    for symbol, ps in st.pairs.items():
        pairs.append({
            "symbol": symbol,
            "last_price": ps.last_price,
            "last_signal_short": ps.last_signal_short,
            "last_signal_mid": ps.last_signal_mid,
            "last_signal_long": ps.last_signal_long,
            "trend_score": ps.trend_score,
            "vol_score": ps.vol_score,
            "meta": ps.meta,
        })

    return {
        "mode": st.mode,
        "trading_enabled": st.trading_enabled,
        "total_realized_pnl": st.total_realized_pnl,
        "total_unrealized_pnl": st.total_unrealized_pnl,
        "accounts": accounts,
        "pairs": pairs,
        "meta": st.meta,
        "scorecard": st.meta.get("scorecard"),
    }
