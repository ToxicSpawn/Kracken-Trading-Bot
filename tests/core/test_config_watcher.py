from __future__ import annotations

import os
from pathlib import Path

from core.config_watcher import ConfigWatcher
from core.policy import PolicyEngine
from core.state import AccountState, GlobalState


def write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_reload_accounts_updates_risk_and_drawdown(tmp_path: Path) -> None:
    accounts_path = tmp_path / "accounts.yaml"
    write_yaml(
        accounts_path,
        """
accounts:
  paper:
    max_drawdown_pct: -15
    risk_multiplier: 1.0
""",
    )

    state = GlobalState(accounts={"paper": AccountState(name="paper", max_drawdown_pct=-20.0, risk_multiplier=0.5)})
    policy = PolicyEngine.from_yaml({"policies": {}})
    watcher = ConfigWatcher(state, policy, accounts_path=str(accounts_path), policies_path=str(tmp_path / "policies.yaml"))

    watcher.reload_once()

    acct = state.accounts["paper"]
    assert acct.max_drawdown_pct == -15.0
    assert acct.risk_multiplier == 1.0

    # update file and ensure second reload picks changes
    write_yaml(
        accounts_path,
        """
accounts:
  paper:
    max_drawdown_pct: -10
    risk_multiplier: 2.5
""",
    )
    current_mtime = os.path.getmtime(accounts_path)
    os.utime(accounts_path, (current_mtime + 1, current_mtime + 1))

    watcher.reload_once()

    acct = state.accounts["paper"]
    assert acct.max_drawdown_pct == -10.0
    assert acct.risk_multiplier == 2.5


def test_reload_policies_replaces_policy_config(tmp_path: Path) -> None:
    policies_path = tmp_path / "policies.yaml"
    write_yaml(
        policies_path,
        """
policies:
  max_daily_loss_aud: -100
  absolute_max_drawdown_pct: -25
  max_single_position_pct: 0.3
  per_symbol_max_loss_aud: -200
""",
    )

    state = GlobalState()
    initial_policy = PolicyEngine.from_yaml({"policies": {"max_daily_loss_aud": -50}})
    watcher = ConfigWatcher(state, initial_policy, accounts_path=str(tmp_path / "accounts.yaml"), policies_path=str(policies_path))

    watcher.reload_once()

    cfg = initial_policy.config
    assert cfg.max_daily_loss_aud == -100.0
    assert cfg.absolute_max_drawdown_pct == -25.0
    assert cfg.max_single_position_pct == 0.3
    assert cfg.per_symbol_max_loss_aud == -200.0
