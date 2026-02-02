#!/bin/bash
# Monitor training output on training PC
# Usage: ./monitor-training.sh [lines]

LINES=${1:-50}
REPO="/home/ooousay/repos/isaac_drone_racer"

echo "=== Last $LINES lines of training output ==="
ssh training-pc "wsl bash -c \"tmux capture-pane -t training -p 2>/dev/null | tail -$LINES\"" 2>/dev/null || \
    ssh training-pc "wsl bash -c \"tail -$LINES $REPO/training.log 2>/dev/null\"" || \
    echo "No training session or log found"
