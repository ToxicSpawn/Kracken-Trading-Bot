import argparse
import asyncio
import os

from core.replay import load_journal_events
from core.global_state import get_global_state
from core.policy import PolicyEngine
from core.runtime import MultiAgentRuntime


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True, help="Path to JSONL journal file")
    args = ap.parse_args()

    os.environ["BOT_MODE"] = "replay"

    state = get_global_state()
    state.mode = "replay"

    with open("config/policies.yaml", "r", encoding="utf-8") as f:
        import yaml
        pol_cfg = yaml.safe_load(f)
    policy = PolicyEngine.from_yaml(pol_cfg)
    rt = MultiAgentRuntime(state, policy)
    await rt.start()

    # Minimal replay: re-submit recorded intents into the intent bus.
    # (Extend later to replay prices/candles deterministically.)
    events = load_journal_events(args.journal)
    for e in events:
        if e.kind == "intent":
            state.submit_intent(e.payload)

    # Run briefly to drain queue
    await asyncio.sleep(2.0)
    await rt.stop()


if __name__ == "__main__":
    asyncio.run(main())

