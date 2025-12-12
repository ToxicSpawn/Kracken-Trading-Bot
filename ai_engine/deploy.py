import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from .config import BASE_DIR, WRITABLE_DIRS, CANDIDATES_DIR, BOT_SERVICE_NAME

logger = logging.getLogger(__name__)


def _is_writable(path: Path) -> bool:
    path = path.resolve()
    for allowed in WRITABLE_DIRS:
        allowed = Path(allowed).resolve()
        if allowed in path.parents or path == allowed:
            return True
    return False


def save_candidates(updated_files: List[Dict[str, Any]], cycle_dir: Path) -> None:
    for f in updated_files:
        rel_path = f.get("path")
        content = f.get("content", "")
        if not rel_path or not content:
            continue

        dest = (cycle_dir / rel_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        logger.info("Saved candidate file: %s", dest)


def apply_candidates(cycle_dir: Path, dry_run: bool = True) -> None:
    for dest in cycle_dir.rglob("*.py"):
        rel = dest.relative_to(cycle_dir)
        live_path = (BASE_DIR / rel).resolve()

        if not _is_writable(live_path):
            logger.warning("Skipping non-writable target: %s", live_path)
            continue

        if dry_run:
            logger.info("[DRY RUN] Would overwrite: %s", live_path)
        else:
            live_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, live_path)
            logger.info("Overwrote live file: %s", live_path)


def restart_bot_service() -> None:
    try:
        subprocess.run(["systemctl", "restart", BOT_SERVICE_NAME], check=True)
        logger.info("Restarted service %s", BOT_SERVICE_NAME)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to restart service %s: %s", BOT_SERVICE_NAME, exc)
