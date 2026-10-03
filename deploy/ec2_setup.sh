#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 22.04 / 24.04 EC2 instance.
# Run from the repo root as the normal login user:   bash deploy/ec2_setup.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE=quantbot

sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git chrony sqlite3
# Signed requests are rejected if our clock is >60s off server time.
sudo systemctl enable --now chrony

cd "$REPO_DIR"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
mkdir -p state logs data

if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  echo ">>> Created .env: add ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY before starting the service."
fi

sed -e "s#__REPO_DIR__#${REPO_DIR}#g" -e "s#__USER__#$(whoami)#g" deploy/quantbot.service \
  | sudo tee /etc/systemd/system/${SERVICE}.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE}

cat <<MSG

Setup complete. Next steps:
  1. nano .env                                  # add API keys
  2. .venv/bin/quantbot check                   # connectivity + account
  3. .venv/bin/quantbot universe                # map Roostoo pairs to Binance symbols
  4. .venv/bin/quantbot smoke-test --trade      # tiny real round-trips; prints accounting setting
  5. set live.dry_run in config/config.yaml, then:
     sudo systemctl start ${SERVICE}
     journalctl -u ${SERVICE} -f
MSG
