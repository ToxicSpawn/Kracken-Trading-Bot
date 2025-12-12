Kraken ML Bot – v200E (Complete Dry-Run Edition)
===============================================

This is the **v200E** version of the Kraken ML bot. It extends v180E with:

- Multi-agent architecture (market, short/mid/long, risk, execution, portfolio, meta)
- Multi-horizon RSI strategy (5m / 15m / 1h)
- Trend & volatility scoring (TrendVolStrategy)
- Volatility-aware sizing in the execution agent
- PolicyEngine with daily loss guardrail, per-symbol loss caps, and drawdown circuit breaker
- Telegram alerts (signals + daily PnL)
- Prometheus metrics exporter
- Data-quality monitor for market feeds (Prometheus gauges + Telegram gap/staleness alerts)
- Hot-reload watcher for policy/account configs (no restart needed for threshold tweaks)
- File-based rotating logs
- FastAPI dashboard for live state
- systemd service + helper scripts
- Regime-aware microstructure overlays (liquidity/volatility/slippage) for sizing + guardrails

> Default mode is **DRY RUN / paper**. Only switch to `BOT_MODE=live` after testing.

Quick start (dry run)
---------------------

1. Upload the zip to your server and unzip:

   unzip kraken_ml_bot_v200E_full.zip -d /opt/kraken_ml_bot_v200E
   cd /opt/kraken_ml_bot_v200E

2. Create virtualenv and install dependencies:

   ./scripts/create_venv.sh

   The helper script now defaults to using the preinstalled system packages to
   avoid network failures in restricted environments. If you need to force
   dependency downloads, run with `INSTALL_DEPS=1 ./scripts/create_venv.sh`.

3. Configure environment:

   cp .env.example .env
   nano .env   # fill Kraken keys, Telegram, BOT_MODE=paper

4. Run the bot (dry run):

   ./scripts/run_bot.sh

Optional dashboard
------------------

In another shell:

   source venv/bin/activate
   uvicorn dashboard.app:app --host 0.0.0.0 --port 8080

Then open:

   http://YOUR_SERVER_IP:8080/
   http://YOUR_SERVER_IP:8080/api/state

Prometheus metrics (if enabled in config/metrics.yaml):

   http://YOUR_SERVER_IP:8001/metrics

Data-quality monitoring (config/monitoring.yaml):

   - Enables gap/staleness checks on market data.
   - Exposes Prometheus gauges `data_feed_gap` and `data_feed_stale` per symbol/timeframe.
   - Sends Telegram alerts when gaps persist beyond thresholds (rate-limited).

Hot reload for configs (accounts + policies):

   - The bot watches `config/accounts.yaml` and `config/policies.yaml` for changes every 30 seconds.
   - Risk multipliers, drawdown caps, and policy thresholds update live—no restart required.

Logs are written to:

   logs/bot.log

Microstructure + regime overlays (new)
--------------------------------------

- MarketDataAgent now tags each pair with:
  - **Regime label:** trending / mean-reverting / sideways / high-volatility.
  - **Liquidity + slippage heuristics:** derived from recent OHLCV depth/volatility.
- PolicyEngine uses these overlays to block trading during event blackout hours, extreme vol,
  thin liquidity, or oversized slippage; ExecutionAgent dynamically resizes or skips orders based on
  regime and stress multipliers.

Backtesting harness (new)
-------------------------

Run a quick RSI+ML backtest over historical candles to validate policy changes before going live:

```
source venv/bin/activate
python scripts/run_backtest.py path/to/ohlcv.csv --symbol XBTUSDT --initial 10000 --fee-bps 10 --slippage-bps 5
```

The CSV must include columns `timestamp, open, high, low, close, volume`. The backtester reuses the
production strategy and reports trade count, win rate, total return, and max drawdown.
