from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Callable

import json

from analytics.backtester import Backtester
from analytics.research.optimizer import SimpleGridOptimizer, ParamResult


@dataclass
class ResearchConfig:
    data_path: Path = Path("data/sol_aud_5m.csv")
    horizon: int = 5
    max_tests: int = 50
    output_path: Path = Path("research") / "best_params.json"


class AutoResearchEngine:
    """
    v900000 Auto-Research Engine

    - Runs backtests with different strategy parameters.
    - Scores them (e.g. Sharpe - DD penalty).
    - Saves best config to a JSON file so the runtime/Omega can consume it.
    """

    def __init__(self, cfg: ResearchConfig | None = None) -> None:
        self.cfg = cfg or ResearchConfig()
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _score_result(self, result: Dict[str, Any]) -> float:
        """
        Expect result to contain at least:
            - sharpe
            - max_drawdown_pct
        You can adapt this based on your Backtester result structure.
        """
        sharpe = float(result.get("sharpe", 0.0))
        max_dd = float(result.get("max_drawdown_pct", 0.0))
        # Example: Sharpe minus penalty for large drawdowns
        return sharpe - max(0.0, (-max_dd - 10.0) / 10.0)

    def run_grid_search(self, param_grid: Dict[str, List[Any]]) -> ParamResult:
        bt = Backtester(data_path=self.cfg.data_path, horizon=self.cfg.horizon)
        optimizer = SimpleGridOptimizer()

        def evaluate(params: Dict[str, Any]) -> float:
            result = bt.run(params)
            return self._score_result(result)

        best = optimizer.search(param_grid, evaluate)
        self._save_best(best)
        return best

    def _save_best(self, best: ParamResult) -> None:
        payload = {
            "params": best.params,
            "score": best.score,
        }
        self.cfg.output_path.write_text(json.dumps(payload, indent=2))
