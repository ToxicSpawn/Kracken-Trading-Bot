from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from core.state import GlobalState
from core.policy import PolicyEngine


logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    state: GlobalState
    policy: PolicyEngine


class Agent(Protocol):
    name: str
    ctx: AgentContext

    async def step(self) -> None:
        ...

    async def run_loop(self) -> None:
        ...

    def stop(self) -> None:
        ...


class BaseAgent:
    def __init__(self, name: str, ctx: AgentContext, interval: float = 1.0) -> None:
        self.name = name
        self.ctx = ctx
        self.interval = interval
        self._running = True

    async def step(self) -> None:  # type: ignore[override]
        raise NotImplementedError

    async def run_loop(self) -> None:  # type: ignore[override]
        logger.info("Agent %s loop started", self.name)
        while self._running:
            try:
                await self.step()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error in agent %s: %s", self.name, exc)
            await asyncio.sleep(self.interval)

    def stop(self) -> None:  # type: ignore[override]
        self._running = False
