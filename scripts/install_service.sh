#!/usr/bin/env bash
set -e

APP_DIR="/opt/kraken_ml_bot_v200E"

echo "[*] Creating application directory at $APP_DIR"
mkdir -p "$APP_DIR"

echo "[*] Copying bot files..."
rsync -a --delete "./" "$APP_DIR/"

cd "$APP_DIR"

echo "[*] Creating virtualenv..."
python3 -m venv venv
source venv/bin/activate

echo "[*] Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f ".env" ]; then
  echo "[!] No .env found in $APP_DIR – please create and configure it."
fi

SERVICE_PATH="/etc/systemd/system/kraken_ml_bot_v200E.service"
echo "[*] Installing systemd service to $SERVICE_PATH"
cp "deploy/kraken_ml_bot_v200E.service" "$SERVICE_PATH"

echo "[*] Reloading systemd..."
systemctl daemon-reload
systemctl enable kraken_ml_bot_v200E.service
systemctl restart kraken_ml_bot_v200E.service

echo "[*] Installation complete. Check logs with:"
echo "    journalctl -u kraken_ml_bot_v200E.service -f"
