from __future__ import annotations

import pytest

from analytics.backtester import BacktestResult, SimpleBacktester
from strategies.base import SignalType, StrategySignal


class StubStrategy:
    def __init__(self, signals: list[SignalType]):
        self.signals = signals
        self.idx = 0

    def generate_signal(self, ohlcv: list[list]) -> StrategySignal:  # noqa: ARG002
        sig = self.signals[min(self.idx, len(self.signals) - 1)]
        self.idx += 1
        return StrategySignal(symbol="TEST", signal=sig, confidence=1.0)


def test_backtester_executes_trades_and_reports_metrics() -> None:
    prices = [100, 110, 90, 120]
    ohlcv = [[i, p, p, p, p, 1_000] for i, p in enumerate(prices)]
    strategy = StubStrategy([SignalType.BUY, SignalType.HOLD, SignalType.HOLD, SignalType.SELL])

    bt = SimpleBacktester(initial_equity=1_000.0, fee_bps=0.0, slippage_bps=0.0)
    result = bt.run(ohlcv, strategy)

    assert isinstance(result, BacktestResult)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == 100
    assert trade.exit_price == 120
    assert trade.pnl == 200.0
    assert result.total_return_pct == pytest.approx(20.0)
    assert result.win_rate == 1.0
    assert result.max_drawdown_pct < 0


def test_backtester_handles_fees_and_slippage() -> None:
    prices = [50, 55, 52, 60]
    ohlcv = [[i, p, p, p, p, 500] for i, p in enumerate(prices)]
    strategy = StubStrategy([SignalType.BUY, SignalType.HOLD, SignalType.SELL, SignalType.HOLD])

    bt = SimpleBacktester(initial_equity=1_000.0, fee_bps=10.0, slippage_bps=10.0)
    result = bt.run(ohlcv, strategy)

    assert len(result.trades) == 1
    trade = result.trades[0]
    # Ensure slippage adjusted prices are used
    assert trade.entry_price == 50 * 1.001
    assert trade.exit_price == 52 * 0.999
    assert trade.pnl < (52 - 50) * (1_000 / 50)  # fees + slippage reduce pnl
    assert result.total_return_pct < 4.0
