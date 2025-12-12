from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class SignalType(Enum):
    HOLD = auto()
    BUY = auto()
    SELL = auto()


@dataclass
class StrategySignal:
    symbol: str
    signal: SignalType
    confidence: float = 0.0
