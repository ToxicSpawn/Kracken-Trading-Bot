import asyncio
import logging
from typing import List

import yaml

from core.state import GlobalState
from core.policy import PolicyEngine
from core.allocator import compute_risk_multiplier, AllocatorConfig
from core.live_metrics import on_intent, on_intent_dropped, on_allocator, on_eligibility
from agents.base import Agent, AgentContext
from agents.market_data import MarketDataAgent
from agents.short_term_agent import ShortTermAgent
from agents.mid_term_agent import MidTermAgent
from agents.long_term_agent import LongTermAgent
from agents.risk_agent import RiskAgent
from agents.execution_agent import ExecutionAgent
from agents.portfolio_agent import PortfolioAgent
from agents.meta_controller import MetaController
from agents.ml_agent import MlAgent
from agents.hedge_agent import HedgeAgent
from agents.arbitrage_agent import ArbitrageAgent
from agents.router_agent import RouterAgent
from agents.market_selection_agent import MarketSelectionAgent
from agents.reasoning_layer import ReasoningLayerAgent
from agents.research_agent import ResearchAgent
from agents.multi_venue_book_agent import MultiVenueBookAgent
from agents.loss_cluster_supervisor import LossClusterSupervisor


logger = logging.getLogger(__name__)


class MultiAgentRuntime:
    def __init__(self, state: GlobalState, policy: PolicyEngine) -> None:
        self.state = state
        self.policy = policy
        self.agents: List[Agent] = []
        self._tasks: List[asyncio.Task] = []
        self._stop = False

    async def start(self) -> None:
        ctx = AgentContext(state=self.state, policy=self.policy)

        with open("config/agents.yaml", "r", encoding="utf-8") as f:
            agent_cfg = yaml.safe_load(f).get("agents", {})

        with open("config/pairs.yaml", "r", encoding="utf-8") as pf:
            pair_cfg = yaml.safe_load(pf).get("pairs", [])
        symbols = [p["symbol"] for p in pair_cfg if p.get("enabled", False)]

        if agent_cfg.get("market_data", True):
            self.agents.append(MarketDataAgent("market_data", ctx, symbols, interval=5.0))
        if agent_cfg.get("short_term", True):
            self.agents.append(ShortTermAgent("short_term", ctx, symbols, interval=5.0))
        if agent_cfg.get("mid_term", True):
            self.agents.append(MidTermAgent("mid_term", ctx, symbols, interval=15.0))
        if agent_cfg.get("long_term", True):
            self.agents.append(LongTermAgent("long_term", ctx, symbols, interval=60.0))
        if agent_cfg.get("risk", True):
            self.agents.append(RiskAgent("risk", ctx, interval=10.0))
        if agent_cfg.get("multi_venue_book"):
            self.agents.append(MultiVenueBookAgent("multi_venue_book", ctx, symbols=symbols, interval=3.0))
        if agent_cfg.get("execution", True):
            self.agents.append(ExecutionAgent("execution", ctx, interval=10.0))
        if agent_cfg.get("portfolio", True):
            self.agents.append(PortfolioAgent("portfolio", ctx, interval=30.0))
        if agent_cfg.get("meta_controller", True):
            self.agents.append(MetaController("meta_controller", ctx, interval=20.0))
        if agent_cfg.get("market_selection"):
            self.agents.append(MarketSelectionAgent("market_selection", ctx, interval=600.0))
        if agent_cfg.get("reasoning_layer"):
            self.agents.append(ReasoningLayerAgent("reasoning_layer", ctx, interval=30.0))
        if agent_cfg.get("ml", False):
            self.agents.append(MlAgent("ml", ctx, symbols, interval=60.0))
        if agent_cfg.get("hedge", False):
            self.agents.append(HedgeAgent("hedge", ctx, symbols, interval=300.0))
        if agent_cfg.get("router", False):
            self.agents.append(RouterAgent("router", ctx, interval=60.0))
        if agent_cfg.get("research"):
            self.agents.append(ResearchAgent("research", ctx, interval=3600.0))
        if agent_cfg.get("arbitrage"):
            # Start conservatively with a single symbol, higher spread threshold,
            # slower cadence, and tiny notional per leg. Increase gradually once
            # live behavior looks stable and clean.
            arb_symbols = ["SOL/AUD"]
            self.agents.append(
                ArbitrageAgent(
                    "arbitrage",
                    ctx=ctx,
                    symbols=arb_symbols,
                    interval=20.0,
                    min_net_spread_bps=50.0,
                    notional_per_trade=2.0,
                )
            )
        if agent_cfg.get("loss_cluster_supervisor", False):
            self.agents.append(LossClusterSupervisor("loss_cluster_supervisor", ctx, interval=2.0))

        for agent in self.agents:
            logger.info("Starting agent: %s", agent.name)
            task = asyncio.create_task(agent.run_loop())
            self._tasks.append(task)

        # Start the governed execution loop
        exec_task = asyncio.create_task(self._governed_execution_loop())
        self._tasks.append(exec_task)

    async def _governed_execution_loop(self) -> None:
        """Global capital allocator applied at the execution choke point.
        
        Processes intents from the queue, applies allocator, and routes to execution.
        This is the single source of truth for position sizing.
        """
        while not self._stop:
            if not self.state.trading_enabled:
                await asyncio.sleep(0.25)
                continue

            # Drain intents (micro-batching)
            intents = self.state.drain_intents(max_n=100)
            if not intents:
                await asyncio.sleep(0.25)
                continue

            for intent in intents:
                # Track intent observed
                on_intent(intent)

                # Basic eligibility check: ensure required fields exist
                if not intent.get("symbol") or intent.get("qty", 0.0) == 0.0:
                    logger.debug("ExecutionAgent: skipping invalid intent %s", intent)
                    on_intent_dropped("missing_fields")
                    continue

                # Apply global capital allocator (choke point)
                alloc = compute_risk_multiplier(self.state, intent, AllocatorConfig())
                intent["qty"] = float(intent.get("qty", 0.0)) * float(alloc.multiplier)
                intent.setdefault("meta", {})["allocator"] = alloc.reasons
                on_allocator(intent)

                # Track eligibility multiplier
                on_eligibility(intent, alloc.multiplier)

                # Skip if allocator reduced qty to zero or below
                if float(intent.get("qty", 0.0)) <= 0.0:
                    logger.debug(
                        "ExecutionAgent: allocator blocked intent %s (mult=%.3f)",
                        intent.get("symbol"),
                        alloc.multiplier,
                    )
                    on_intent_dropped("allocator_zeroed")
                    continue

                # Risk policy check (if policy engine available)
                policy_decision = self.policy.evaluate(self.state)
                if not policy_decision.can_trade:
                    logger.debug(
                        "ExecutionAgent: policy blocked intent %s: %s",
                        intent.get("symbol"),
                        "; ".join(policy_decision.reasons),
                    )
                    on_intent_dropped("policy_blocked")
                    continue

                # Intent is approved - log for now (actual execution happens in ExecutionAgent)
                logger.info(
                    "ExecutionAgent: approved intent %s qty=%.6f (alloc_mult=%.3f)",
                    intent.get("symbol"),
                    intent.get("qty"),
                    alloc.multiplier,
                )
                # Store approved intent in meta for ExecutionAgent to pick up
                # or route directly to execution if needed
                approved_intents = self.state.meta.get("approved_intents", [])
                approved_intents.append(intent)
                self.state.meta["approved_intents"] = approved_intents[-10:]  # Keep last 10

    async def stop(self) -> None:
        self._stop = True
        for a in self.agents:
            a.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
