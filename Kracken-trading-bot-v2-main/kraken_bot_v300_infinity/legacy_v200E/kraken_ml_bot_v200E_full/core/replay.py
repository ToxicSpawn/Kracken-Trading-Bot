from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ReplayEvent:
    ts: str
    kind: str
    payload: Dict[str, Any]


def load_journal_events(path: str) -> List[ReplayEvent]:
    events: List[ReplayEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            events.append(ReplayEvent(ts=obj.get("ts", ""), kind=obj.get("kind", ""), payload=obj.get("payload", {})))
    return events


def iter_events_by_kind(events: Iterable[ReplayEvent], kind: str) -> Iterable[ReplayEvent]:
    for e in events:
        if e.kind == kind:
            yield e

