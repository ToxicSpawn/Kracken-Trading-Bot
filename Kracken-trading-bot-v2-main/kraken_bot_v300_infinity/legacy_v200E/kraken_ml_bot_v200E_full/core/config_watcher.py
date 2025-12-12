from __future__ import annotations

"""Config watcher for hot-reloading configuration.

Compatibility note:
- Uses Optional[X] instead of X | None for Python 3.9 compatibility.
"""

import asyncio
import logging
import os
from typing import Dict, Optional

import yaml

from core.policy import PolicyEngine
from core.state import AccountState, GlobalState

logger = logging.getLogger(__name__)


class ConfigWatcher:
    """Lightweight watcher that hot-reloads accounts and policy configs."""

    def __init__(
        self,
        state: GlobalState,
        policy_engine: PolicyEngine,
        *,
        accounts_path: str = "config/accounts.yaml",
        policies_path: str = "config/policies.yaml",
        interval_sec: float = 30.0,
    ) -> None:
        self.state = state
        self.policy_engine = policy_engine
        self.accounts_path = accounts_path
        self.policies_path = policies_path
        self.interval_sec = interval_sec
        self._mtimes: Dict[str, float] = {}
        self._stopped = False

    def _yaml_if_changed(self, path: str) -> Optional[Dict]:
        if not os.path.exists(path):
            return None

        mtime = os.path.getmtime(path)
        if self._mtimes.get(path) == mtime:
            return None

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self._mtimes[path] = mtime
        return cfg

    def _reload_accounts(self) -> None:
        cfg = self._yaml_if_changed(self.accounts_path)
        if cfg is None:
            return

        accounts_cfg = cfg.get("accounts", {})
        for name, acct_cfg in accounts_cfg.items():
            acct = self.state.accounts.setdefault(name, AccountState(name=name))
            acct.max_drawdown_pct = float(acct_cfg.get("max_drawdown_pct", acct.max_drawdown_pct))
            acct.risk_multiplier = float(acct_cfg.get("risk_multiplier", acct.risk_multiplier))

        logger.info("Hot-reloaded accounts config for %s accounts", len(accounts_cfg))

    def _reload_policies(self) -> None:
        cfg = self._yaml_if_changed(self.policies_path)
        if cfg is None:
            return

        new_engine = PolicyEngine.from_yaml(cfg)
        self.policy_engine.config = new_engine.config
        logger.info("Hot-reloaded policy config")

    def reload_once(self) -> None:
        self._reload_accounts()
        self._reload_policies()

    async def watch_loop(self) -> None:
        while not self._stopped:
            try:
                self.reload_once()
                await asyncio.sleep(self.interval_sec)
            except asyncio.CancelledError:
                self._stopped = True
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Config watcher error: %s", exc)

    def stop(self) -> None:
        self._stopped = True
