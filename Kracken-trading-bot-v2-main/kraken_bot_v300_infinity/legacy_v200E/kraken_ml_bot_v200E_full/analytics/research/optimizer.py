from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class ParamResult:
    params: Dict[str, Any]
    score: float


class SimpleGridOptimizer:
    """
    v120000 simple grid-style optimizer for strategy params.

    You can plug this into your nightly pipeline to test a handful of RSI/MACD/
    ML thresholds and store best results.
    """

    def search(
        self,
        param_grid: Dict[str, List[Any]],
        evaluate_fn: Callable[[Dict[str, Any]], float],
    ) -> ParamResult:
        keys = list(param_grid.keys())
        best = ParamResult(params={}, score=-1e9)

        def _rec(idx: int, current: Dict[str, Any]) -> None:
            nonlocal best
            if idx == len(keys):
                score = evaluate_fn(current)
                if score > best.score:
                    best = ParamResult(params=dict(current), score=score)
                return
            k = keys[idx]
            for v in param_grid[k]:
                current[k] = v
                _rec(idx + 1, current)

        _rec(0, {})
        return best
