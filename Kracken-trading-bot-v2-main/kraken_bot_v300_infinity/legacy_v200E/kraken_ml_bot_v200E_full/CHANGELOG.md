Changelog
=========

v200E (this build)
------------------
- Multi-agent runtime (market, short/mid/long, risk, execution, portfolio, meta).
- RSI + TrendVolStrategy scoring to adjust size.
- PolicyEngine with daily loss guardrail.
- Enhanced short/mid momentum stack: RSI + MACD + Bollinger blend with
  dynamic confidence and regime-aware trend/volatility scoring.
- Telegram alerts for trades and daily PnL.
- Prometheus metrics server.
- FastAPI dashboard exposing global state.
- systemd service & scripts for VPS usage.
