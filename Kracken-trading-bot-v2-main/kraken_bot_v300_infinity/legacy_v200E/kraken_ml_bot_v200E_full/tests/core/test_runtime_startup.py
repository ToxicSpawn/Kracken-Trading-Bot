import asyncio
from pathlib import Path


def build_config(tmp_path: Path, agents_cfg: str, pairs_cfg: str) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agents.yaml").write_text(agents_cfg)
    (config_dir / "pairs.yaml").write_text(pairs_cfg)
    return config_dir


class FakeAgent:
    def __init__(self, name, ctx, symbols=None, interval: float = 0.0):  # noqa: ANN001
        self.name = name
        self.ctx = ctx
        self.symbols = symbols
        self.interval = interval
        self.started = False
        self.stopped = False

    async def run_loop(self):  # noqa: D401
        """Fake run loop that exits immediately."""
        self.started = True
        await asyncio.sleep(0)

    def stop(self):  # noqa: D401
        """Mark agent as stopped."""
        self.stopped = True


def test_runtime_includes_enabled_agents(monkeypatch, tmp_path):
    from core.policy import PolicyConfig, PolicyEngine
    from core.runtime import MultiAgentRuntime
    from core.state import GlobalState

    agents_cfg = """
    agents:
      market_data: true
      short_term: false
      mid_term: true
      long_term: false
      risk: true
      execution: false
      portfolio: true
      meta_controller: true
    """
    pairs_cfg = """
    pairs:
      - symbol: "AAA/BBB"
        enabled: true
      - symbol: "CCC/DDD"
        enabled: false
      - symbol: "EEE/FFF"
        enabled: true
    """

    config_dir = build_config(tmp_path, agents_cfg, pairs_cfg)
    monkeypatch.chdir(config_dir.parent)

    from core import runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "MarketDataAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "ShortTermAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "MidTermAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "LongTermAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "RiskAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "ExecutionAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "PortfolioAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "MetaController", FakeAgent)

    async def run_runtime():
        state = GlobalState()
        policy = PolicyEngine(PolicyConfig(-200.0, -20.0, 0.4))
        runtime = MultiAgentRuntime(state, policy)

        await runtime.start()

        created_names = {agent.name for agent in runtime.agents}
        assert created_names == {"market_data", "mid_term", "risk", "portfolio", "meta_controller"}

        for agent in runtime.agents:
            if agent.symbols is not None:
                assert agent.symbols == ["AAA/BBB", "EEE/FFF"]

        await runtime.stop()

    asyncio.run(run_runtime())


def test_runtime_handles_no_enabled_pairs(monkeypatch, tmp_path):
    from core.policy import PolicyConfig, PolicyEngine
    from core.runtime import MultiAgentRuntime
    from core.state import GlobalState

    agents_cfg = """
    agents:
      market_data: true
      short_term: true
      mid_term: true
      long_term: true
      risk: true
      execution: true
      portfolio: true
      meta_controller: true
    """
    pairs_cfg = """
    pairs:
      - symbol: "AAA/BBB"
        enabled: false
      - symbol: "CCC/DDD"
        enabled: false
    """

    config_dir = build_config(tmp_path, agents_cfg, pairs_cfg)
    monkeypatch.chdir(config_dir.parent)

    from core import runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "MarketDataAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "ShortTermAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "MidTermAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "LongTermAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "RiskAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "ExecutionAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "PortfolioAgent", FakeAgent)
    monkeypatch.setattr(runtime_mod, "MetaController", FakeAgent)

    async def run_runtime():
        state = GlobalState()
        policy = PolicyEngine(PolicyConfig(-200.0, -20.0, 0.4))
        runtime = MultiAgentRuntime(state, policy)

        await runtime.start()

        symbol_consuming = [a for a in runtime.agents if a.symbols is not None]
        assert symbol_consuming
        for agent in symbol_consuming:
            assert agent.symbols == []

        await runtime.stop()

    asyncio.run(run_runtime())
