#!/usr/bin/env bash
set -e

# Cloud deployment script for Kraken Trading Bot
# This script automates the installation process on a cloud server

APP_DIR="/opt/kraken_ml_bot_v200E"
SERVICE_NAME="kraken_ml_bot_v200E.service"
CURRENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================="
echo "Kraken Trading Bot - Cloud Deployment"
echo "========================================="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script requires sudo privileges. Please run with sudo."
    exit 1
fi

# Check Python version
echo "[*] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "[!] Error: Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "[✓] Python version OK: $PYTHON_VERSION"

# Update system packages
echo ""
echo "[*] Updating system packages..."
apt update && apt install -y python3-pip python3-venv git curl build-essential python3-dev libgomp1 || {
    echo "[!] Failed to update packages. Continuing anyway..."
}

# Create application directory
echo ""
echo "[*] Creating application directory at $APP_DIR"
mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR/logs"

# Copy files
echo "[*] Copying bot files from $CURRENT_DIR to $APP_DIR"
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='venv' \
    --exclude='*.log' "$CURRENT_DIR/" "$APP_DIR/" || {
    # If rsync fails, try cp
    echo "[*] rsync not available, using cp..."
    cp -r "$CURRENT_DIR"/* "$APP_DIR/" 2>/dev/null || true
    cp -r "$CURRENT_DIR"/.env.example "$APP_DIR/" 2>/dev/null || true
}

cd "$APP_DIR"

# Create virtual environment
echo ""
echo "[*] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "[*] Virtual environment already exists, skipping..."
else
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "[*] Upgrading pip..."
pip install --upgrade pip --quiet || {
    echo "[!] Warning: Failed to upgrade pip. Continuing..."
}

# Install dependencies
echo ""
echo "[*] Installing Python dependencies..."
echo "[*] This may take a few minutes..."

# Try to install all dependencies
pip install -r requirements.txt || {
    echo "[!] Warning: Some dependencies may have failed to install."
    echo "[*] Attempting to install critical dependencies individually..."
    pip install ccxt aiohttp fastapi uvicorn pydantic python-dotenv PyYAML numpy pandas scipy prometheus-client jinja2 || true
}

# Check for .env file
echo ""
if [ ! -f ".env" ]; then
    echo "[!] No .env file found. Creating template..."
    cat > .env << 'EOF'
# Exchange API Keys
KRAKEN_API_KEY=your_api_key_here
KRAKEN_API_SECRET=your_api_secret_here

# Bot Configuration
BOT_MODE=paper
LOG_LEVEL=INFO

# Telegram Notifications (optional)
# TELEGRAM_BOT_TOKEN=your_telegram_bot_token
# TELEGRAM_CHAT_ID=your_telegram_chat_id

# Live Metrics (optional)
ENABLE_METRICS=true
METRICS_HOST=0.0.0.0
METRICS_PORT=9109

# OpenAI (optional)
# OPENAI_API_KEY=your_openai_key
EOF
    echo "[!] IMPORTANT: Please edit $APP_DIR/.env and configure your API keys!"
    echo "[!] Set BOT_MODE=paper for testing (default)."
    chmod 600 .env
else
    echo "[✓] .env file found"
    chmod 600 .env
fi

# Install systemd service
echo ""
echo "[*] Installing systemd service..."

# Update service file paths if needed
if [ -f "deploy/$SERVICE_NAME" ]; then
    cp "deploy/$SERVICE_NAME" "/etc/systemd/system/$SERVICE_NAME"
    
    # Ensure paths in service file are correct
    sed -i "s|WorkingDirectory=.*|WorkingDirectory=$APP_DIR|g" "/etc/systemd/system/$SERVICE_NAME"
    sed -i "s|EnvironmentFile=.*|EnvironmentFile=$APP_DIR/.env|g" "/etc/systemd/system/$SERVICE_NAME"
    sed -i "s|ExecStart=.*|ExecStart=$APP_DIR/venv/bin/python $APP_DIR/bot.py|g" "/etc/systemd/system/$SERVICE_NAME"
    
    echo "[✓] Service file installed"
else
    # Create service file if it doesn't exist
    cat > "/etc/systemd/system/$SERVICE_NAME" << EOF
[Unit]
Description=Kraken ML Bot v200E
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/python $APP_DIR/bot.py
Restart=on-failure
RestartSec=10
User=root
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    echo "[✓] Service file created"
fi

# Reload systemd
echo "[*] Reloading systemd..."
systemctl daemon-reload

# Enable service (but don't start yet - user should configure .env first)
systemctl enable "$SERVICE_NAME" || {
    echo "[!] Warning: Failed to enable service"
}

# Set permissions
echo ""
echo "[*] Setting file permissions..."
chown -R root:root "$APP_DIR" 2>/dev/null || true
chmod +x "$APP_DIR/bot.py" 2>/dev/null || true
chmod +x "$APP_DIR/scripts"/*.sh 2>/dev/null || true

# Summary
echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "Installation directory: $APP_DIR"
echo "Service name: $SERVICE_NAME"
echo ""
echo "Next steps:"
echo "1. Edit configuration: sudo nano $APP_DIR/.env"
echo "   - Add your Kraken API keys"
echo "   - Set BOT_MODE=paper for testing"
echo ""
echo "2. Test the installation:"
echo "   cd $APP_DIR"
echo "   source venv/bin/activate"
echo "   python bot.py"
echo ""
echo "3. Start the service (after configuring .env):"
echo "   sudo systemctl start $SERVICE_NAME"
echo ""
echo "4. Check status:"
echo "   sudo systemctl status $SERVICE_NAME"
echo ""
echo "5. View logs:"
echo "   sudo journalctl -u $SERVICE_NAME -f"
echo "   # or"
echo "   tail -f $APP_DIR/logs/bot.log"
echo ""
echo "⚠️  IMPORTANT: Start with BOT_MODE=paper for testing!"
echo ""

