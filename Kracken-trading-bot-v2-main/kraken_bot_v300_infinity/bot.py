from __future__ import annotations

import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# v300 entry point delegates to the hardened v200E runtime for now.
BASE_DIR = Path(__file__).resolve().parent
LEGACY_DIR = BASE_DIR / "legacy_v200E" / "kraken_ml_bot_v200E_full"

if not LEGACY_DIR.exists():
    raise SystemExit(f"Legacy runtime not found at {LEGACY_DIR}")

# Ensure imports resolve even if launched from a different CWD.
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

os.chdir(LEGACY_DIR)

from core.app import main  # type: ignore[import]  # noqa: E402


def _setup_logging() -> None:
    log_dir = LEGACY_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "bot.log"

    handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[handler, logging.StreamHandler()],
    )


async def run() -> None:
    _setup_logging()
    logging.getLogger(__name__).info("Starting v300 entry (delegating to v200E core)")
    await main()


if __name__ == "__main__":
    asyncio.run(run())