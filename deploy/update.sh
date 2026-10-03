#!/usr/bin/env bash
# Pull the latest code, run the tests, and restart the bot only if they pass.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
git pull --ff-only
.venv/bin/pip install -e ".[dev]" -q
.venv/bin/pytest -q
sudo systemctl restart quantbot
sleep 3
systemctl status quantbot --no-pager -l | head -20
