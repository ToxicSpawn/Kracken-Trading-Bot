from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from strategies.base import SignalType, StrategySignal


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    pnl: float


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: List[float]
    total_return_pct: float
    win_rate: float
    max_drawdown_pct: float


class SimpleBacktester:
    """Lightweight, long-only backtester that reuses existing strategies.

    The harness feeds each candle to a strategy that exposes a
    ``generate_signal`` method. It executes at most one fully invested
    position at a time, applies configurable fees + slippage, and returns
    core diagnostics such as win rate and max drawdown.
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self.initial_equity = float(initial_equity)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)

    def _apply_fill_adjustment(self, price: float, is_entry: bool) -> float:
        adj = self.slippage_bps / 10_000
        return price * (1 + adj if is_entry else 1 - adj)

    def _trade_fee(self, notional: float) -> float:
        return notional * (self.fee_bps / 10_000)

    def _mark_to_market(self, cash: float, position_size: float, price: float) -> float:
        return cash + (position_size * price)

    def _max_drawdown_pct(self, equity_curve: Sequence[float]) -> float:
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            drawdown = ((eq - peak) / peak) * 100
            if drawdown < max_dd:
                max_dd = drawdown
        return max_dd

    def run(self, ohlcv: List[list], strategy: object) -> BacktestResult:
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        cash = self.initial_equity
        position_size = 0.0
        entry_price = 0.0
        trades: List[Trade] = []
        equity_curve: List[float] = []

        for idx, row in df.iterrows():
            history = df.iloc[: idx + 1].values.tolist()
            signal_obj = strategy.generate_signal(history)
            if not isinstance(signal_obj, StrategySignal):
                raise TypeError("Strategy must return StrategySignal")

            price = float(row["close"])

            if signal_obj.signal is SignalType.BUY and position_size == 0:
                fill_price = self._apply_fill_adjustment(price, is_entry=True)
                size = cash / fill_price if fill_price else 0.0
                fee = self._trade_fee(fill_price * size)
                cash -= (fill_price * size) + fee
                position_size = size
                entry_price = fill_price
            elif signal_obj.signal is SignalType.SELL and position_size > 0:
                exit_price = self._apply_fill_adjustment(price, is_entry=False)
                proceeds = exit_price * position_size
                fee = self._trade_fee(proceeds)
                cash += proceeds - fee
                pnl = (exit_price - entry_price) * position_size - (
                    self._trade_fee(entry_price * position_size) + fee
                )
                trades.append(
                    Trade(
                        symbol=signal_obj.symbol,
                        entry_time=pd.to_datetime(row["timestamp"] if idx > 0 else df.iloc[0]["timestamp"]),
                        exit_time=pd.to_datetime(row["timestamp"]),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=position_size,
                        pnl=pnl,
                    )
                )
                position_size = 0.0
                entry_price = 0.0

            equity_curve.append(self._mark_to_market(cash, position_size, price))

        if position_size > 0:
            # Mark final equity with last close for an open position.
            equity_curve[-1] = self._mark_to_market(cash, position_size, df.iloc[-1]["close"])

        total_return_pct = ((equity_curve[-1] / self.initial_equity) - 1) * 100
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = wins / len(trades) if trades else 0.0
        max_drawdown_pct = self._max_drawdown_pct(equity_curve) if equity_curve else 0.0

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            max_drawdown_pct=max_drawdown_pct,
        )


class Backtester:
    """Lightweight wrapper used by research routines.

    It loads price data from CSV and computes simple metrics so the grid search
    can rank parameter combinations. This intentionally avoids strategy
    coupling; plug in your preferred simulation details here.
    """

    def __init__(self, data_path: Path, horizon: int = 5) -> None:
        self.data_path = Path(data_path)
        self.horizon = int(horizon)

    def run(self, params: Dict[str, Any]) -> Dict[str, float]:
        if not self.data_path.exists():
            return {"sharpe": 0.0, "max_drawdown_pct": 0.0}

        df = pd.read_csv(self.data_path)
        close_col = "close" if "close" in df.columns else df.columns[-1]
        close = df[close_col].astype(float)

        returns = close.pct_change(self.horizon).dropna()
        if returns.empty or returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = (returns.mean() / returns.std()) * (len(returns) ** 0.5)

        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        max_dd = float(((equity - peak) / peak * 100).min()) if not equity.empty else 0.0

        return {"sharpe": float(sharpe), "max_drawdown_pct": max_dd}
