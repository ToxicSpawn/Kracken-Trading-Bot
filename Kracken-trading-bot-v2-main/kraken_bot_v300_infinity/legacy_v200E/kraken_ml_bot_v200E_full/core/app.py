import asyncio
import logging
import os
from pathlib import Path
from typing import Iterable

import yaml

from core.global_state import get_global_state
from core.state import AccountState
from core.runtime import MultiAgentRuntime
from core.policy import PolicyEngine
from core.utils import utcnow
from core.config_watcher import ConfigWatcher
from core.app_supervisor import HealthRule, monitor_health
from metrics.server import start_metrics_server
from reports.daily_pnl import schedule_daily_pnl_task

logger = logging.getLogger(__name__)


def _require_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _require_files(paths: Iterable[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required config files: {', '.join(missing)}")


def _preflight_validate(pairs_cfg: dict, accounts: dict, policy: PolicyEngine) -> None:
    enabled = [p for p in pairs_cfg.get("pairs", []) if p.get("enabled", False)]
    if not enabled:
        raise RuntimeError("Preflight failed: no enabled pairs in config/pairs.yaml")

    if not accounts:
        raise RuntimeError("Preflight failed: no accounts loaded from config/accounts.yaml")

    # Sanity: policy must have evaluate() and/or loaded thresholds
    if not hasattr(policy, "evaluate"):
        raise RuntimeError("Preflight failed: PolicyEngine missing evaluate()")


def _validate_live_env(mode: str) -> None:
    if mode != "live":
        return

    required = [
        "KRAKEN_API_KEY",
        "KRAKEN_API_SECRET",
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET",
        "COINBASE_API_PASSPHRASE",
    ]
    missing = [env for env in required if not os.getenv(env)]
    if missing:
        raise RuntimeError(
            f"Live mode requested but these env vars are missing: {', '.join(sorted(missing))}"
        )

    if os.getenv("I_UNDERSTAND_LIVE_TRADING") != "1":
        raise RuntimeError(
            "Set I_UNDERSTAND_LIVE_TRADING=1 to acknowledge live-trading risk before starting in live mode."
        )


async def main() -> None:
    state = get_global_state()
    state.mode = os.getenv("BOT_MODE", "paper")

    _require_file("config/accounts.yaml")
    _require_file("config/pairs.yaml")
    _require_file("config/policies.yaml")
    _require_file("config/agents.yaml")

    # Load accounts
    with open("config/accounts.yaml", "r", encoding="utf-8") as f:
        acct_cfg = yaml.safe_load(f)

    for a in acct_cfg.get("accounts", []):
        name = a.get("name", "default")
        state.accounts[name] = AccountState(
            name=name,
            equity=float(a.get("equity", 0.0)),
            balance=float(a.get("balance", 0.0)),
        )

    # Load pairs (for preflight)
    with open("config/pairs.yaml", "r", encoding="utf-8") as f:
        pairs_cfg = yaml.safe_load(f)

    # Policy engine
    with open("config/policies.yaml", "r", encoding="utf-8") as f:
        pol_cfg = yaml.safe_load(f)
    policy_engine = PolicyEngine.from_yaml(pol_cfg)

    # Preflight validation (fail fast)
    _preflight_validate(pairs_cfg, state.accounts, policy_engine)

    # Start metrics (optional)
    try:
        with open("config/metrics.yaml", "r", encoding="utf-8") as f:
            metrics_cfg = yaml.safe_load(f).get("metrics", {})
        if metrics_cfg.get("enabled", True):
            port = int(metrics_cfg.get("port", 8001))
            start_metrics_server(port)
            logger.info("Prometheus metrics running on port %s", port)
    except Exception:
        logger.exception("Metrics server failed to start (continuing)")

    # Daily PnL task (optional)
    try:
        with open("config/reports.yaml", "r", encoding="utf-8") as f:
            reports_cfg = yaml.safe_load(f).get("reports", {})
        daily_cfg = reports_cfg.get("daily_pnl", {})
        if daily_cfg.get("enabled", True):
            schedule_daily_pnl_task(
                hour_utc=int(daily_cfg.get("hour_utc", 23)),
                minute_utc=int(daily_cfg.get("minute_utc", 55))
            )
    except Exception:
        logger.exception("Daily PnL scheduler failed to start (continuing)")

    # Hot reload watchers
    cfg_watcher = ConfigWatcher(state, policy_engine)
    watcher_task = asyncio.create_task(cfg_watcher.watch_loop(), name="config_watcher")

    # Runtime
    runtime = MultiAgentRuntime(state, policy_engine)
    await runtime.start()

    # Health supervisor (authoritative)
    rules = [
        HealthRule("market_data", max_staleness_seconds=20, severity="CRITICAL"),
        HealthRule("risk", max_staleness_seconds=30, severity="CRITICAL"),
        HealthRule("execution_intent", max_staleness_seconds=60, severity="WARN"),
    ]
    health_task = asyncio.create_task(monitor_health(state, rules, interval_seconds=5.0), name="health_monitor")

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Shutdown requested")
    finally:
        cfg_watcher.stop()
        for t in (watcher_task, health_task):
            t.cancel()
        await asyncio.gather(watcher_task, health_task, return_exceptions=True)
        await runtime.stop()
