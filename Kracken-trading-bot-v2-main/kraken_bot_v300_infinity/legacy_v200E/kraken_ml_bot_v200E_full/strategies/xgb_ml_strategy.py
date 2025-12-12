from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
try:
    from joblib import load
except ImportError:  # pragma: no cover - optional dependency
    load = None

from strategies.base import StrategySignal, SignalType
from strategies.trend_vol_strategy import TrendVolStrategy


@dataclass
class MlConfig:
    model_path: Path
    horizon: int = 5  # how many candles ahead the model was trained on
    threshold: float = 0.55  # probability required to take a directional trade


class XgbMlStrategy:
    """
    XGBoost-based ML strategy.

    - Loads a pre-trained model from disk.
    - Builds a feature vector from OHLCV + trend/vol scores.
    - Produces BUY/SELL/HOLD based on predicted probability of an
      upward move over the chosen horizon.

    This does NOT replace your RSI/TrendVol strategies – it is an
    additional alpha source that you can blend inside ExecutionAgent.
    """

    def __init__(self, symbol: str, cfg: MlConfig) -> None:
        self.symbol = symbol
        self.cfg = cfg
        self._model = self._load_model()
        self._trend_vol = TrendVolStrategy(symbol)

    def _load_model(self):
        if not self.cfg.model_path.exists():
            # Fail safe: no model -> HOLD forever.
            return None
        if load is None:
            print("[XgbMlStrategy] joblib not installed; ML model disabled.")
            return None
        try:
            return load(self.cfg.model_path)
        except Exception as exc:  # noqa: BLE001
            # If loading fails, run in "disabled" mode.
            print(f"[XgbMlStrategy] Failed to load model at {self.cfg.model_path}: {exc}")
            return None

    def _build_features(self, ohlcv: List[list]) -> Optional[np.ndarray]:
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        if len(close) < 60:
            return None

        # Basic returns
        returns = close.pct_change().fillna(0.0)
        ret_1 = returns.iloc[-1]
        ret_5 = (close.iloc[-1] / close.iloc[-5]) - 1.0
        ret_10 = (close.iloc[-1] / close.iloc[-10]) - 1.0

        # Volatility measures
        vol_10 = returns.rolling(10).std().iloc[-1]
        vol_20 = returns.rolling(20).std().iloc[-1]

        # Volume stats
        vol_mean_20 = volume.rolling(20).mean().iloc[-1]
        vol_last = volume.iloc[-1]
        vol_ratio = (vol_last / vol_mean_20) if vol_mean_20 > 0 else 1.0

        # Trend/vol scores (your existing helper)
        trend_score, vol_score = self._trend_vol.score(ohlcv)

        # Recent momentum / skew
        ret_window = returns.tail(20)
        skew = ret_window.skew()
        kurt = ret_window.kurt()

        features = np.array(
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
        ).reshape(1, -1)

        return features

    def generate_signal(self, ohlcv: List[list]) -> StrategySignal:
        if self._model is None:
            return StrategySignal(symbol=self.symbol, signal=SignalType.HOLD, confidence=0.0)

        features = self._build_features(ohlcv)
        if features is None:
            return StrategySignal(symbol=self.symbol, signal=SignalType.HOLD, confidence=0.2)

        # Assume binary classification: class 1 = "up move"
        try:
            proba = self._model.predict_proba(features)[0, 1]
        except Exception as exc:  # noqa: BLE001
            print(f"[XgbMlStrategy] Prediction failed: {exc}")
            return StrategySignal(symbol=self.symbol, signal=SignalType.HOLD, confidence=0.0)

        if proba >= self.cfg.threshold:
            sig = SignalType.BUY
            conf = float(min(0.99, max(0.3, proba)))
        elif proba <= (1.0 - self.cfg.threshold):
            sig = SignalType.SELL
            conf = float(min(0.99, max(0.3, 1.0 - proba)))
        else:
            sig = SignalType.HOLD
            conf = float(0.25)

        return StrategySignal(symbol=self.symbol, signal=sig, confidence=conf)
