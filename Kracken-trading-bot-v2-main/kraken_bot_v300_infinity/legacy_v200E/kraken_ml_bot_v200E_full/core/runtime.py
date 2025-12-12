import asyncio
import logging
from typing import List, Optional, Any

import yaml

from core.state import GlobalState
from core.policy import PolicyEngine
from agents.base import Agent, AgentContext

from agents.market_data import MarketDataAgent
from agents.risk_agent import RiskAgent
from agents.meta_controller import MetaController
from agents.router_agent import RouterAgent
from agents.market_selection_agent import MarketSelectionAgent
from agents.reasoning_layer import ReasoningLayerAgent
from agents.research_agent import ResearchAgent
from agents.multi_venue_book_agent import MultiVenueBookAgent
from agents.portfolio_agent import PortfolioAgent

# Level-14: Intent-based agents (these may need to be created)
try:
    from agents.execution_intent_agent import ExecutionIntentAgent
    from agents.hedge_intent_agent import HedgeIntentAgent
    from agents.arbitrage_intent_agent import ArbitrageIntentAgent
except ImportError:
    # Placeholder - these agents need to be created
    ExecutionIntentAgent = None
    HedgeIntentAgent = None
    ArbitrageIntentAgent = None

# Level-14: Risk governance and adapters (these may need to be created)
try:
    from core.risk_governor import RiskGovernor
    from adapters.simulator import SimulatedAdapter
    from adapters.kraken import KrakenAdapter
    from core.reconcile import Reconciler, ReconcileConfig
except ImportError:
    # Placeholder - these need to be created
    RiskGovernor = None
    SimulatedAdapter = None
    KrakenAdapter = None
    Reconciler = None
    ReconcileConfig = None

from core.intent_netting import net_intents, NettingConfig


logger = logging.getLogger(__name__)


class MultiAgentRuntime:
    """Level-14 runtime.

    Adds:
    - intent netting / compression before execution
    - journal hooks preserved (if journal present)
    - relies on GlobalState heartbeat + system_mode
    """

    def __init__(self, state: GlobalState, policy: PolicyEngine, journal: Optional[Any] = None) -> None:
        self.state = state
        self.policy = policy
        self.journal = journal

        self.agents: List[Agent] = []
        self._tasks: List[asyncio.Task] = []
        self._stop = False

        # Risk governor (if available)
        if RiskGovernor:
            self.risk_governor = RiskGovernor(state=self.state, policy_engine=self.policy, journal=self.journal)
        else:
            self.risk_governor = None
            logger.warning("RiskGovernor not available - risk checks will be skipped")

        mode = getattr(self.state, "mode", "shadow")
        if mode in ("shadow", "backtest", "sim", "replay"):
            if SimulatedAdapter:
                self.exchange = SimulatedAdapter(state=self.state)
            else:
                logger.warning("SimulatedAdapter not available - using None")
                self.exchange = None
            self._reconcile_enabled = False
        else:
            if KrakenAdapter:
                self.exchange = KrakenAdapter()
            else:
                logger.warning("KrakenAdapter not available - using None")
                self.exchange = None
            self._reconcile_enabled = True

        # Reconciler (if available)
        if Reconciler and self.exchange:
            self.reconciler = Reconciler(
                state=self.state,
                adapter=self.exchange,
                journal=self.journal,
                config=ReconcileConfig(interval_seconds=20, fills_lookback_seconds=120, drift_threshold=3) if ReconcileConfig else None,
            )
        else:
            self.reconciler = None

        self._net_cfg = NettingConfig(max_batch=250, keep_sides_separate=False)

    async def start(self) -> None:
        ctx = AgentContext(state=self.state, policy=self.policy)

        with open("config/agents.yaml", "r", encoding="utf-8") as f:
            agent_cfg = yaml.safe_load(f).get("agents", {})

        legacy_direct = [k for k in ("execution", "hedge", "arbitrage") if agent_cfg.get(k, False)]
        legacy_signal = [k for k in ("short_term", "mid_term", "long_term", "ml") if agent_cfg.get(k, False)]
        if legacy_direct:
            raise RuntimeError("Direct-execution agents are disabled in Level-14: " + ", ".join(legacy_direct))
        if legacy_signal:
            raise RuntimeError("Duplicate-signal agents are disabled in Level-14: " + ", ".join(legacy_signal))

        with open("config/pairs.yaml", "r", encoding="utf-8") as pf:
            pair_cfg = yaml.safe_load(pf).get("pairs", [])
        symbols = [p["symbol"] for p in pair_cfg if p.get("enabled", False)]

        if agent_cfg.get("market_data", True):
            self.agents.append(MarketDataAgent("market_data", ctx, symbols, interval=5.0))
        if agent_cfg.get("multi_venue_book", True):
            self.agents.append(MultiVenueBookAgent("multi_venue_book", ctx, symbols=symbols, interval=3.0))
        if agent_cfg.get("risk", True):
            self.agents.append(RiskAgent("risk", ctx, interval=10.0))
        if agent_cfg.get("portfolio", True):
            self.agents.append(PortfolioAgent("portfolio", ctx, interval=30.0))
        if agent_cfg.get("router", True):
            self.agents.append(RouterAgent("router", ctx, interval=60.0))
        if agent_cfg.get("market_selection", True):
            self.agents.append(MarketSelectionAgent("market_selection", ctx, interval=600.0))
        if agent_cfg.get("reasoning_layer", True):
            self.agents.append(ReasoningLayerAgent("reasoning_layer", ctx, interval=30.0))
        if agent_cfg.get("meta_controller", True):
            self.agents.append(MetaController("meta_controller", ctx, interval=20.0))
        if agent_cfg.get("research", True):
            self.agents.append(ResearchAgent("research", ctx, interval=3600.0))

        if agent_cfg.get("execution_intent", True) and ExecutionIntentAgent:
            self.agents.append(ExecutionIntentAgent("execution_intent", ctx, symbols=symbols, interval=10.0))
        if agent_cfg.get("hedge_intent", False) and HedgeIntentAgent:
            self.agents.append(HedgeIntentAgent("hedge_intent", ctx, symbols=symbols, interval=300.0))
        if agent_cfg.get("arbitrage_intent", False) and ArbitrageIntentAgent:
            self.agents.append(
                ArbitrageIntentAgent(
                    "arbitrage_intent",
                    ctx=ctx,
                    symbols=["SOL/AUD"],
                    interval=20.0,
                    min_net_spread_bps=50.0,
                    notional_per_trade=2.0,
                )
            )

        for agent in self.agents:
            logger.info("Starting agent: %s", agent.name)
            self._tasks.append(asyncio.create_task(agent.run_loop(), name=f"agent:{agent.name}"))

        self._tasks.append(asyncio.create_task(self._governed_execution_loop(), name="governed_execution"))
        if self._reconcile_enabled and self.reconciler:
            self._tasks.append(asyncio.create_task(self.reconciler.run_forever(), name="reconciler"))

    async def _governed_execution_loop(self) -> None:
        while not self._stop:
            if getattr(self.state, "system_mode", "NORMAL") == "PAUSED":
                await asyncio.sleep(1.0)
                continue

            # Drain and net intents (micro-batching)
            intents = self.state.drain_intents(max_n=self._net_cfg.max_batch)
            if not intents:
                await asyncio.sleep(0.25)
                continue

            for intent in net_intents(intents, self._net_cfg):
                if getattr(self.state, "system_mode", "NORMAL") == "MANAGE_ONLY" and intent.get("exposure_delta", 0.0) > 0:
                    continue

                if self.risk_governor:
                    decision = self.risk_governor.evaluate(intent)
                    if not decision.allow:
                        continue

                if self.exchange:
                    try:
                        await self.exchange.place_order(intent)
                    except Exception as e:
                        logger.exception("Order submission failed: %r intent=%s", e, intent)
                        self.state.set_system_mode("DEGRADED", reason="EXECUTION_ERRORS")
                        await asyncio.sleep(1.0)

    async def stop(self) -> None:
        self._stop = True
        for a in self.agents:
            a.stop()
        try:
            if self.reconciler:
                self.reconciler.stop()
        except Exception:
            pass
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
