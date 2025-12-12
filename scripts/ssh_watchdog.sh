#!/usr/bin/env bash
set -e

echo "[*] Checking sshd status..."
if ! systemctl is-active --quiet ssh && ! systemctl is-active --quiet sshd; then
  echo "[!] SSH service is not active. Attempting restart..."
  if systemctl list-unit-files | grep -q '^ssh\.service'; then
    systemctl restart ssh
  elif systemctl list-unit-files | grep -q '^sshd\.service'; then
    systemctl restart sshd
  fi
  echo "[*] SSH service restart attempted."
else
  echo "[*] SSH service is running."
fi
