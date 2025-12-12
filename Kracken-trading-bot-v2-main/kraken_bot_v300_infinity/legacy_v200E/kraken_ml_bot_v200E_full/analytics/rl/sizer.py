from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class RlSizerConfig:
    min_multiplier: float = 0.2
    max_multiplier: float = 3.0
    step: float = 0.2
    gamma: float = 0.95          # discount factor
    lr: float = 0.05             # learning rate
    epsilon: float = 0.1         # exploration probability
    state_bins: int = 7          # granularity for bucketing


class RlPositionSizer:
    """
    Simple tabular RL sizer (v12000).

    - State: (win_rate_bucket, drawdown_bucket, regime_bucket)
    - Action: discrete multiplier in [min_multiplier, max_multiplier] steps.
    - Reward: recent PnL delta (clipped).

    This is deliberately small & robust. You can later swap this for a
    neural net if you want to go even crazier.
    """

    def __init__(self, cfg: Optional[RlSizerConfig] = None) -> None:
        self.cfg = cfg or RlSizerConfig()
        self.actions = np.arange(
            self.cfg.min_multiplier,
            self.cfg.max_multiplier + 1e-9,
            self.cfg.step,
            dtype=np.float32,
        )
        # Q-table: state_idx -> np.array(len(actions))
        self.q: Dict[int, np.ndarray] = {}
        self._last_state_idx: Optional[int] = None
        self._last_action_idx: Optional[int] = None

    def _bucketize(self, value: float, low: float, high: float) -> int:
        v = max(low, min(high, value))
        norm = (v - low) / (high - low + 1e-9)
        return int(norm * (self.cfg.state_bins - 1))

    def _encode_state(
        self,
        win_rate: float,      # [0, 1]
        drawdown_pct: float,  # [-100, 0]
        regime_score: float,  # [-1, 1]
    ) -> int:
        wr_bin = self._bucketize(win_rate, 0.0, 1.0)
        dd_bin = self._bucketize(drawdown_pct, -100.0, 0.0)
        reg_bin = self._bucketize(regime_score, -1.0, 1.0)
        return wr_bin * 100 + dd_bin * 10 + reg_bin

    def _ensure_state(self, s_idx: int) -> None:
        if s_idx not in self.q:
            self.q[s_idx] = np.zeros(len(self.actions), dtype=np.float32)

    def choose_multiplier(
        self,
        win_rate: float,
        drawdown_pct: float,
        regime_score: float,
    ) -> float:
        s_idx = self._encode_state(win_rate, drawdown_pct, regime_score)
        self._ensure_state(s_idx)

        # epsilon-greedy exploration
        if np.random.rand() < self.cfg.epsilon:
            a_idx = np.random.randint(len(self.actions))
        else:
            a_idx = int(np.argmax(self.q[s_idx]))

        self._last_state_idx = s_idx
        self._last_action_idx = a_idx
        return float(self.actions[a_idx])

    def update(
        self,
        reward: float,
        next_win_rate: float,
        next_drawdown_pct: float,
        next_regime_score: float,
    ) -> None:
        if self._last_state_idx is None or self._last_action_idx is None:
            return

        r = float(np.clip(reward, -5.0, 5.0))
        s_idx = self._last_state_idx
        a_idx = self._last_action_idx

        next_s_idx = self._encode_state(next_win_rate, next_drawdown_pct, next_regime_score)
        self._ensure_state(next_s_idx)

        best_next = float(np.max(self.q[next_s_idx]))
        td_target = r + self.cfg.gamma * best_next
        td_error = td_target - self.q[s_idx][a_idx]

        self.q[s_idx][a_idx] += self.cfg.lr * td_error

        self._last_state_idx = None
        self._last_action_idx = None
