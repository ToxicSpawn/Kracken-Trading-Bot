# Cloud Server Deployment Guide

This guide will help you deploy the Kraken Trading Bot to a cloud server (Ubuntu/Debian).

## Prerequisites

- A cloud server running Ubuntu 20.04+ or Debian 11+
- SSH access with sudo privileges
- Python 3.10+ installed on the server
- Git (optional, if cloning from repository)

## Quick Deployment

### Option 1: Using the Automated Install Script (Recommended)

1. **Transfer files to your server:**
   ```bash
   # On your local machine, create a tarball
   tar czf kraken_bot.tar.gz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='venv' .
   
   # Transfer to server (replace USER@SERVER with your details)
   scp kraken_bot.tar.gz USER@SERVER:/tmp/
   ```

2. **SSH into your server:**
   ```bash
   ssh USER@SERVER
   ```

3. **Run the deployment script:**
   ```bash
   cd /tmp
   tar xzf kraken_bot.tar.gz
   cd Kracken-trading-bot  # or whatever the extracted directory is named
   chmod +x scripts/deploy_cloud.sh
   sudo ./scripts/deploy_cloud.sh
   ```

### Option 2: Manual Installation

Follow these steps if you prefer manual installation:

#### Step 1: Prepare the Server

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and essential tools
sudo apt install -y python3 python3-pip python3-venv git curl

# Create application directory
sudo mkdir -p /opt/kraken_ml_bot_v200E
sudo chown $USER:$USER /opt/kraken_ml_bot_v200E
```

#### Step 2: Transfer and Extract Bot Files

```bash
# Transfer files (from your local machine)
scp -r . USER@SERVER:/opt/kraken_ml_bot_v200E/

# Or clone from git if available
# cd /opt/kraken_ml_bot_v200E
# git clone <repository-url> .
```

#### Step 3: Install Dependencies

```bash
cd /opt/kraken_ml_bot_v200E

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### Step 4: Configure Environment

```bash
# Create .env file
cp .env.example .env  # if available, or create manually
nano .env
```

Required environment variables:
```env
# Exchange API Keys
KRAKEN_API_KEY=your_api_key_here
KRAKEN_API_SECRET=your_api_secret_here

# Bot Configuration
BOT_MODE=paper  # Use 'paper' for testing, 'live' for production
LOG_LEVEL=INFO

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Live Metrics (optional)
ENABLE_METRICS=true
METRICS_HOST=0.0.0.0
METRICS_PORT=9109

# OpenAI (if using ML features)
OPENAI_API_KEY=your_openai_key  # optional
```

**⚠️ Important:** Start with `BOT_MODE=paper` for testing!

#### Step 5: Create Logs Directory

```bash
mkdir -p logs
```

#### Step 6: Test the Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Run bot in test mode (Ctrl+C to stop)
python bot.py
```

If it starts without errors, proceed to set up as a service.

#### Step 7: Install as Systemd Service

```bash
# Copy service file
sudo cp deploy/kraken_ml_bot_v200E.service /etc/systemd/system/

# Edit service file if needed (adjust paths)
sudo nano /etc/systemd/system/kraken_ml_bot_v200E.service

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable kraken_ml_bot_v200E.service

# Start the service
sudo systemctl start kraken_ml_bot_v200E.service

# Check status
sudo systemctl status kraken_ml_bot_v200E.service

# View logs
sudo journalctl -u kraken_ml_bot_v200E.service -f
```

## Post-Deployment

### Accessing Services

- **Bot Logs:** `tail -f /opt/kraken_ml_bot_v200E/logs/bot.log`
- **Systemd Logs:** `sudo journalctl -u kraken_ml_bot_v200E.service -f`
- **Dashboard:** `http://YOUR_SERVER_IP:8080` (if enabled)
- **Metrics:** `http://YOUR_SERVER_IP:9109/metrics` (if ENABLE_METRICS=true)

### Managing the Service

```bash
# Start bot
sudo systemctl start kraken_ml_bot_v200E.service

# Stop bot
sudo systemctl stop kraken_ml_bot_v200E.service

# Restart bot
sudo systemctl restart kraken_ml_bot_v200E.service

# Check status
sudo systemctl status kraken_ml_bot_v200E.service

# Disable auto-start on boot
sudo systemctl disable kraken_ml_bot_v200E.service
```

### Firewall Configuration

If your cloud provider uses a firewall (UFW, iptables, or cloud firewall), open necessary ports:

```bash
# For metrics endpoint (if enabled)
sudo ufw allow 9109/tcp

# For dashboard (if enabled)
sudo ufw allow 8080/tcp

# For existing metrics (if enabled)
sudo ufw allow 8001/tcp
```

### Security Considerations

1. **Protect API Keys:** Ensure `.env` file has restricted permissions:
   ```bash
   chmod 600 /opt/kraken_ml_bot_v200E/.env
   ```

2. **Use Firewall:** Only expose necessary ports
3. **Monitor Logs:** Regularly check for errors or unusual activity
4. **Backup Configuration:** Keep backups of your `.env` and `config/` directory

## Troubleshooting

### Bot won't start

1. Check logs: `sudo journalctl -u kraken_ml_bot_v200E.service -n 50`
2. Verify Python version: `python3 --version` (needs 3.10+)
3. Check virtual environment: `source venv/bin/activate && python --version`
4. Verify dependencies: `pip list | grep -E "(ccxt|aiohttp|fastapi)"`

### Service fails to start

1. Check file paths in service file
2. Verify `.env` file exists and is readable
3. Check file permissions: `ls -la /opt/kraken_ml_bot_v200E/`
4. Test manually: `cd /opt/kraken_ml_bot_v200E && source venv/bin/activate && python bot.py`

### Dependencies fail to install

1. Update pip: `pip install --upgrade pip`
2. Install build tools: `sudo apt install -y build-essential python3-dev`
3. For XGBoost: `sudo apt install -y libgomp1`

### Connection issues

1. Verify network connectivity
2. Check firewall rules
3. Verify API keys are correct
4. Test API connection manually

## Updating the Bot

```bash
# Stop the service
sudo systemctl stop kraken_ml_bot_v200E.service

# Backup current installation
sudo cp -r /opt/kraken_ml_bot_v200E /opt/kraken_ml_bot_v200E.backup

# Transfer new files (or pull from git)
# ... copy/update files ...

# Update dependencies if requirements.txt changed
cd /opt/kraken_ml_bot_v200E
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl start kraken_ml_bot_v200E.service
```

## Monitoring

Consider setting up:
- **Log rotation:** Already configured in bot.py
- **Health checks:** Monitor the metrics endpoint
- **Alerting:** Use Telegram notifications for critical events
- **Backup:** Regular backups of configuration and logs

