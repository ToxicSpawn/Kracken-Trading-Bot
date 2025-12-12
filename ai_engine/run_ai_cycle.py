#!/usr/bin/env python3
import logging
import os
from datetime import datetime
from pathlib import Path

from .config import AI_ENGINE_ENABLED, AI_ENGINE_DIR, REPORTS_DIR, CANDIDATES_DIR
from .code_collector import collect_code_and_logs
from .ai_client import request_improvements
from .deploy import save_candidates, apply_candidates, restart_bot_service

logging.basicConfig(
    level=os.getenv("AI_ENGINE_LOG_LEVEL", "INFO"),
    format="%(asctime)s | AI_ENGINE | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ai_engine")


def run_cycle(auto_deploy: bool = False, dry_run: bool = True) -> None:
    if not AI_ENGINE_ENABLED:
        logger.warning("AI engine disabled via env (AI_ENGINE_ENABLED=false)")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    cycle_dir = CANDIDATES_DIR / timestamp
    cycle_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting code and logs...")
    context = collect_code_and_logs()

    logger.info("Requesting improvements from AI model...")
    ai_result = request_improvements(context)

    report_path = REPORTS_DIR / f"ai_report_{timestamp}.md"
    report_lines = [
        f"# AI Review {timestamp}",
        "",
        "## Analysis",
        ai_result.get("analysis", ""),
        "",
        "## Risks",
    ]
    for r in ai_result.get("risks", []):
        report_lines.append(f"- {r}")
    report_lines.append("")
    report_lines.append("## Recommended Changes")
    for c in ai_result.get("recommended_changes", []):
        report_lines.append(f"- {c}")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Saved AI report: %s", report_path)

    updated_files = ai_result.get("updated_files", [])
    if not updated_files:
        logger.info("No updated_files provided by AI; nothing to apply.")
        return

    save_candidates(updated_files, cycle_dir)

    if auto_deploy:
        logger.info("Auto-deploy is ON. Applying candidates...")
        apply_candidates(cycle_dir, dry_run=dry_run)
        if not dry_run:
            logger.info("Restarting bot service after deployment...")
            restart_bot_service()
    else:
        logger.info("Auto-deploy is OFF. Review candidates before applying.")


if __name__ == "__main__":
    AUTO_DEPLOY = os.getenv("AI_ENGINE_AUTO_DEPLOY", "false").lower() == "true"
    DRY_RUN = os.getenv("AI_ENGINE_DRY_RUN", "true").lower() == "true"
    run_cycle(auto_deploy=AUTO_DEPLOY, dry_run=DRY_RUN)
