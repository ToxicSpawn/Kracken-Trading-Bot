import os
from pathlib import Path

BASE_DIR = Path(os.getenv("BOT_BASE_DIR", Path(__file__).resolve().parents[1])).resolve()

AI_ENGINE_ENABLED = os.getenv("AI_ENGINE_ENABLED", "true").lower() == "true"
AI_MODEL = os.getenv("AI_ENGINE_MODEL", "gpt-4.1-mini")

CODE_DIRECTORIES = [
    BASE_DIR / "bot.py",
    BASE_DIR / "core",
    BASE_DIR / "agents",
    BASE_DIR / "strategies",
    BASE_DIR / "analytics",
]

LOG_FILES = [
    BASE_DIR / "logs" / "bot.log",
]

AI_ENGINE_DIR = BASE_DIR / "ai_engine"
REPORTS_DIR = AI_ENGINE_DIR / "reports"
CANDIDATES_DIR = AI_ENGINE_DIR / "candidates"

WRITABLE_DIRS = [
    BASE_DIR / "strategies",
    BASE_DIR / "config",
]

BOT_SERVICE_NAME = os.getenv("BOT_SYSTEMD_SERVICE", "kraken_ml_bot_v200E.service")
