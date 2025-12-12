#!/usr/bin/env bash
set -e

# Create a virtualenv that can also see the preinstalled system packages.
python3 -m venv venv --system-site-packages
source venv/bin/activate

# In restricted environments (e.g., when outbound traffic is blocked),
# dependency installation will fail. Default to skipping network installs
# unless explicitly requested via INSTALL_DEPS=1.
if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "INSTALL_DEPS=0 (default): using system packages and skipping pip install."
  echo "Set INSTALL_DEPS=1 to force dependency installation."
fi
