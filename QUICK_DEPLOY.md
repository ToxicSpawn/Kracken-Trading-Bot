# Quick Cloud Deployment Guide

## Step-by-Step Cloud Server Installation

### 1. Prepare Your Files

On your local machine, create a package to transfer:

```bash
# Create a tarball (excludes unnecessary files)
tar czf kraken_bot.tar.gz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='venv' --exclude='*.log' .
```

### 2. Transfer to Cloud Server

```bash
# Replace USER and SERVER_IP with your details
scp kraken_bot.tar.gz USER@SERVER_IP:/tmp/
```

### 3. SSH Into Your Server

```bash
ssh USER@SERVER_IP
```

### 4. Run Automated Installation

```bash
cd /tmp
tar xzf kraken_bot.tar.gz
cd Kracken-trading-bot  # adjust if your directory name differs
sudo chmod +x scripts/deploy_cloud.sh
sudo ./scripts/deploy_cloud.sh
```

### 5. Configure the Bot

```bash
sudo nano /opt/kraken_ml_bot_v200E/.env
```

Add your configuration:
```env
KRAKEN_API_KEY=your_key_here
KRAKEN_API_SECRET=your_secret_here
BOT_MODE=paper
ENABLE_METRICS=true
METRICS_PORT=9109
```

### 6. Test Before Starting Service

```bash
cd /opt/kraken_ml_bot_v200E
source venv/bin/activate
python bot.py
```

Press Ctrl+C after verifying it starts correctly.

### 7. Start as a Service

```bash
sudo systemctl start kraken_ml_bot_v200E.service
sudo systemctl status kraken_ml_bot_v200E.service
```

### 8. Monitor

```bash
# View logs
sudo journalctl -u kraken_ml_bot_v200E.service -f

# Or check log file
tail -f /opt/kraken_ml_bot_v200E/logs/bot.log

# Access metrics (if enabled)
curl http://localhost:9109/metrics
```

## Common Commands

```bash
# Start bot
sudo systemctl start kraken_ml_bot_v200E.service

# Stop bot
sudo systemctl stop kraken_ml_bot_v200E.service

# Restart bot
sudo systemctl restart kraken_ml_bot_v200E.service

# View status
sudo systemctl status kraken_ml_bot_v200E.service

# View logs
sudo journalctl -u kraken_ml_bot_v200E.service -f
```

## Firewall Setup (if needed)

```bash
sudo ufw allow 9109/tcp  # Metrics endpoint
sudo ufw allow 8080/tcp  # Dashboard (if enabled)
```

## Troubleshooting

- **Bot won't start:** Check `sudo journalctl -u kraken_ml_bot_v200E.service -n 50`
- **Missing dependencies:** `cd /opt/kraken_ml_bot_v200E && source venv/bin/activate && pip install -r requirements.txt`
- **Permission errors:** Ensure service user has access to `/opt/kraken_ml_bot_v200E`

## Important Security Notes

1. **Always start with BOT_MODE=paper** for testing
2. Protect your `.env` file: `sudo chmod 600 /opt/kraken_ml_bot_v200E/.env`
3. Use a firewall to restrict access to metrics endpoints
4. Regularly monitor logs for unusual activity

