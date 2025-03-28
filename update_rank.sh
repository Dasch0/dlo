#!/bin/bash
set -e

# Activate virtual environment and run script
source /root/nebulous-ranking/.venv/bin/activate
python /root/nebulous-ranking/rank_clean.py

# Commit and push website changes
cd /root/nebulous-ranking/docs
git add .
git commit -m "Auto-update: $(date +'%Y-%m-%d %H:%M:%S')"
git push origin main
