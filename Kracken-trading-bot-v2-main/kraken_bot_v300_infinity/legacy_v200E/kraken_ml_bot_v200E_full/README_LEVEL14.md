# Level-14 Full Upgrade Patch

This patch adds the highest-leverage production upgrades:
- App lifecycle supervisor (preflight + health enforcement)
- Agent heartbeats in GlobalState
- Intent micro-batching + netting (execution quality)
- Replay scaffolding (journal -> intent replay)
- Unit tests for netting

## Apply
1) Merge folders into your repo root:
   - core/state.py
   - core/runtime.py
   - core/intent_netting.py
   - core/replay.py
   - core/app_supervisor.py
   - core/app.py
   - scripts/run_replay.py
   - tests/test_intent_netting.py

2) Ensure agents call state.heartbeat(self.name) once per loop iteration.
   If your BaseAgent already has a hook, add it there (best).

3) Run:
   pytest -q

4) Start in shadow:
   BOT_MODE=shadow ./scripts/run_bot.sh

## Notes
- This patch does NOT enable live trading.
- Replay mode uses SimulatedAdapter.
- Some dependencies (RiskGovernor, Reconciler, Intent Agents) may need to be created separately.

