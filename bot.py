import asyncio
import logging
import os
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Load .env file at startup
load_dotenv()

from core.live_metrics import start_metrics_server
start_metrics_server()

from core.app import main

LOG_FILE = "logs/bot.log"
os.makedirs("logs", exist_ok=True)

handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[handler, logging.StreamHandler()],
)

if __name__ == "__main__":
    asyncio.run(main())
