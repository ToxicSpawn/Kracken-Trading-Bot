import asyncio
import logging
import os

import yaml

from core.global_state import get_global_state
from core.state import AccountState
from core.runtime import MultiAgentRuntime
from core.policy import PolicyEngine
from core.utils import utcnow
from core.config_watcher import ConfigWatcher
from metrics.server import start_metrics_server
from reports.daily_pnl import schedule_daily_pnl_task

logger = logging.getLogger(__name__)


async def main() -> None:
    state = get_global_state()
    state.mode = os.getenv("BOT_MODE", "paper")

    # Load accounts
    with open("config/accounts.yaml", "r", encoding="utf-8") as f:
        accounts_cfg = yaml.safe_load(f).get("accounts", {})
    for name, cfg in accounts_cfg.items():
        equity = float(cfg.get("equity", 0.0))
        balance = float(cfg.get("balance", equity))
        state.accounts.setdefault(
            name,
            AccountState(
                name=name,
                equity=equity,
                balance=balance,
                max_drawdown_pct=float(cfg.get("max_drawdown_pct", -20.0)),
                risk_multiplier=float(cfg.get("risk_multiplier", 1.0)),
            ),
        )

    # Policy engine
    with open("config/policies.yaml", "r", encoding="utf-8") as f:
        pol_cfg = yaml.safe_load(f)
    policy_engine = PolicyEngine.from_yaml(pol_cfg)

    # Hot-reload watcher
    cfg_watcher = ConfigWatcher(state, policy_engine)
    watcher_task = asyncio.create_task(cfg_watcher.watch_loop())

    # Metrics
    with open("config/metrics.yaml", "r", encoding="utf-8") as f:
        metrics_cfg = yaml.safe_load(f).get("metrics", {})
    if metrics_cfg.get("enabled", True):
        port = int(metrics_cfg.get("port", 8001))
        start_metrics_server(port)
        logger.info("Prometheus metrics running on port %s", port)

    # Daily PnL task
    with open("config/reports.yaml", "r", encoding="utf-8") as f:
        reports_cfg = yaml.safe_load(f).get("reports", {})
    daily_cfg = reports_cfg.get("daily_pnl", {})
    if daily_cfg.get("enabled", True):
        schedule_daily_pnl_task(hour_utc=int(daily_cfg.get("hour_utc", 23)),
                                minute_utc=int(daily_cfg.get("minute_utc", 55)))

    logger.info("Starting v200E runtime at %s, mode=%s", utcnow().isoformat(), state.mode)

    runtime = MultiAgentRuntime(state, policy_engine)
    await runtime.start()

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Shutdown requested")
    finally:
        cfg_watcher.stop()
        watcher_task.cancel()
        await asyncio.gather(watcher_task, return_exceptions=True)
        await runtime.stop()
