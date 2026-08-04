#!/bin/bash
# backup_databases.sh — daily snapshot of both SQLite databases, kept for 14 days.
# Local backup only (protects against accidental deletion / corruption / bad
# code overwriting data). Does NOT protect against full VPS/disk loss — that
# needs an off-site copy, which is a reasonable next step but adds real
# complexity (cloud storage credentials, etc.) — flagging honestly rather
# than pretending this is a complete disaster-recovery solution.

set -e
BACKUP_DIR="/opt/backups"
DATE=$(date +%Y-%m-%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

if [ -f /opt/btc-quant/trades.db ]; then
  sqlite3 /opt/btc-quant/trades.db ".backup '$BACKUP_DIR/trades_$DATE.db'"
fi

if [ -f /opt/btc-research/research.db ]; then
  sqlite3 /opt/btc-research/research.db ".backup '$BACKUP_DIR/research_$DATE.db'"
fi

# also back up the risk state and learned weights — small but important
[ -f /opt/btc-quant/risk_state.json ] && cp /opt/btc-quant/risk_state.json "$BACKUP_DIR/risk_state_$DATE.json"
[ -f /opt/btc-quant/weights.json ] && cp /opt/btc-quant/weights.json "$BACKUP_DIR/weights_$DATE.json"
[ -f /opt/btc-research/config/weights.json ] && cp /opt/btc-research/config/weights.json "$BACKUP_DIR/research_weights_$DATE.json"

# delete anything older than 14 days
find "$BACKUP_DIR" -type f -mtime +14 -delete

echo "Backup complete: $BACKUP_DIR ($(ls "$BACKUP_DIR" | wc -l) files retained)"
