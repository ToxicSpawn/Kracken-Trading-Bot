from __future__ import annotations

import asyncio
import logging

from core.global_state import get_global_state
from core.utils import utcnow
from notifications.telegram import send_telegram_message

logger = logging.getLogger(__name__)


async def send_daily_pnl_report() -> None:
    st = get_global_state()
    text = (
        "📊 DAILY PnL REPORT\n"
        f"Date: {utcnow().date()}\n\n"
        f"Realized PnL: {st.total_realized_pnl:.2f} AUD\n"
        f"Unrealized PnL: {st.total_unrealized_pnl:.2f} AUD\n"
        f"Trading Enabled: {st.trading_enabled}\n"
    )
    await send_telegram_message(text)


def schedule_daily_pnl_task(hour_utc: int, minute_utc: int) -> None:
    async def _runner():
        while True:
            now = utcnow()
            if now.hour == hour_utc and now.minute == minute_utc:
                logger.info("Daily PnL report trigger at %s", now.isoformat())
                await send_daily_pnl_report()
                await asyncio.sleep(60)
            await asyncio.sleep(20)

    asyncio.create_task(_runner())
