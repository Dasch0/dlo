#!/bin/bash
set -e

# Copy battle reports to git repo as a backup
cp -r /srv/steam/.steam/steam/steamapps/common/NEBULOUS\ Dedicated\ Server/Saves/SkirmishReports/*.xml /root/nebulous-ranking/docs/BattleReports/
cp -r /srv/steam/.steam/steam/steamapps/common/NEBULOUS\ Dedicated\ Server/Saves/SkirmishReports/*.bbr /root/nebulous-ranking/docs/replays/

# Activate virtual environment and run script
source /root/nebulous-ranking/.venv/bin/activate
python /root/nebulous-ranking/rank_clean.py

# Commit and push website changes
cd /root/nebulous-ranking/docs
git add .
git commit -m "Auto-update: $(date +'%Y-%m-%d %H:%M:%S')"
git push
