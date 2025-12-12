from pathlib import Path
from typing import List

import logging

from .config import CODE_DIRECTORIES, LOG_FILES

logger = logging.getLogger(__name__)


def _read_file(path: Path, max_chars: int = 20000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read %s: %s", path, exc)
        return ""


def collect_code_and_logs() -> str:
    sections: List[str] = []

    sections.append("=== CODE SNAPSHOT ===")
    for item in CODE_DIRECTORIES:
        p = Path(item)
        if p.is_file():
            sections.append(f"\n\n### FILE: {p}\n")
            sections.append(_read_file(p))
        elif p.is_dir():
            for sub in p.rglob("*.py"):
                sections.append(f"\n\n### FILE: {sub}\n")
                sections.append(_read_file(sub))
        else:
            logger.info("Skipping non-existent path: %s", p)

    sections.append("\n\n=== RECENT LOGS ===")
    for log_path in LOG_FILES:
        p = Path(log_path)
        if p.exists():
            sections.append(f"\n\n### LOG: {p.name}\n")
            sections.append(_read_file(p, max_chars=10000))

    return "\n".join(sections)
