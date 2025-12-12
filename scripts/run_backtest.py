from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analytics.backtester import SimpleBacktester
from strategies.rsi_ml_strategy import RsiMlStrategy


DEFAULT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_ohlcv(csv_path: Path) -> list[list]:
    df = pd.read_csv(csv_path)
    missing_cols = [c for c in DEFAULT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    return df[DEFAULT_COLUMNS].values.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick RSI+ML backtest over OHLCV CSV data.")
    parser.add_argument("csv", type=Path, help="Path to CSV with columns: timestamp, open, high, low, close, volume")
    parser.add_argument("--symbol", default="XBTUSDT", help="Symbol passed to the strategy (default: XBTUSDT)")
    parser.add_argument("--initial", type=float, default=10_000.0, help="Starting equity for the simulation")
    parser.add_argument("--fee-bps", type=float, default=10.0, help="Fee in basis points applied to each trade leg")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage in basis points on entries/exits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ohlcv = load_ohlcv(args.csv)

    strategy = RsiMlStrategy(symbol=args.symbol)
    backtester = SimpleBacktester(
        initial_equity=args.initial,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    result = backtester.run(ohlcv, strategy)

    print("--- Backtest summary ---")
    print(f"Trades: {len(result.trades)} | Win rate: {result.win_rate*100:.1f}%")
    print(f"Return: {result.total_return_pct:.2f}% | Max DD: {result.max_drawdown_pct:.2f}%")
    if result.trades:
        last = result.trades[-1]
        print(
            f"Last trade: entry={last.entry_price:.4f} exit={last.exit_price:.4f} "
            f"pnl={last.pnl:.2f} size={last.size:.6f}"
        )


if __name__ == "__main__":
    main()
