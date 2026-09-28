#!/bin/sh
# Schedule `galata-fetch update` once a day with launchd (macOS), for this checkout.
#
#   scripts/schedule_reference_update.sh              # install, or replace, the daily job at 09:00 local
#   scripts/schedule_reference_update.sh --uninstall  # remove it
#
# Binance publishes a UTC day a few hours after midnight UTC; a day asked too early is recorded
# `absent` and asked again by the next run. GALATA_CONTACT is not passed, so bls.gov is not asked
# and the BLS events already held are kept. The log is var/logs/reference-update.log.
set -eu

LABEL="com.galata.research.reference-update"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO=$(cd "$(dirname "$0")/.." && pwd)
UV=$(command -v uv)

if [ "${1:-}" = "--uninstall" ]; then
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "removed $LABEL"
    exit 0
fi

mkdir -p "$REPO/var/logs" "$(dirname "$PLIST")"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$UV</string><string>run</string><string>--project</string><string>$REPO</string><string>galata-fetch</string><string>update</string></array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>$REPO/var/logs/reference-update.log</string>
  <key>StandardErrorPath</key><string>$REPO/var/logs/reference-update.log</string>
</dict>
</plist>
EOF
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "scheduled $LABEL daily at 09:00: $UV run galata-fetch update (log: var/logs/reference-update.log)"
