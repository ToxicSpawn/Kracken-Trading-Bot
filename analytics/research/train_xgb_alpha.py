from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from strategies.trend_vol_strategy import TrendVolStrategy


def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df[required]


def build_dataset(df: pd.DataFrame, horizon: int = 5) -> tuple[np.ndarray, np.ndarray]:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    returns = close.pct_change().fillna(0.0)

    rows = []
    labels = []

    tv = TrendVolStrategy("TRAIN")

    ohlcv_raw = df.values.tolist()

    for idx in range(60, len(df) - horizon):
        sub_ohlcv = ohlcv_raw[: idx + 1]

        ret_1 = returns.iloc[idx]
        ret_5 = (close.iloc[idx] / close.iloc[idx - 5]) - 1.0
        ret_10 = (close.iloc[idx] / close.iloc[idx - 10]) - 1.0

        ret_slice = returns.iloc[idx - 20 : idx]
        vol_10 = ret_slice.tail(10).std()
        vol_20 = ret_slice.std()

        vol_mean_20 = volume.iloc[idx - 20 : idx].mean()
        vol_last = volume.iloc[idx]
        vol_ratio = (vol_last / vol_mean_20) if vol_mean_20 > 0 else 1.0

        trend_score, vol_score = tv.score(sub_ohlcv)

        skew = ret_slice.skew()
        kurt = ret_slice.kurt()

        f = np.array(
            [
                ret_1,
                ret_5,
                ret_10,
                vol_10,
                vol_20,
                vol_ratio,
                trend_score,
                vol_score,
                float(skew) if np.isfinite(skew) else 0.0,
                float(kurt) if np.isfinite(kurt) else 0.0,
            ],
            dtype=np.float32,
        )

        future_price = close.iloc[idx + horizon]
        curr_price = close.iloc[idx]
        future_ret = (future_price / curr_price) - 1.0

        label = 1 if future_ret > 0 else 0

        rows.append(f)
        labels.append(label)

    X = np.stack(rows)
    y = np.array(labels, dtype=np.int16)
    return X, y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost ML alpha model from OHLCV CSV.")
    p.add_argument("csv", type=Path, help="Path to OHLCV CSV (timestamp, open, high, low, close, volume)")
    p.add_argument("--horizon", type=int, default=5, help="Prediction horizon in candles")
    p.add_argument("--output", type=Path, default=Path("models/xgb_alpha.pkl"), help="Output model path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_ohlcv(args.csv)
    X, y = build_dataset(df, horizon=args.horizon)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("--- Classification report ---")
    print(classification_report(y_test, y_pred))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(model, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
