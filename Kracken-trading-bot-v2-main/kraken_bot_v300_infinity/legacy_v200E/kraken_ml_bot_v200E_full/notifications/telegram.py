from __future__ import annotations

import os
import logging
import aiohttp

logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


async def send_telegram_message(text: str) -> None:
    if not TOKEN or not CHAT_ID:
        logger.debug("Telegram: disabled (missing TOKEN or CHAT_ID).")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as resp:
                if resp.status != 200:
                    logger.warning("Telegram: non-200 response: %s", resp.status)
    except Exception as exc:  # noqa: BLE001
        logger.error("Telegram send error: %s", exc)
