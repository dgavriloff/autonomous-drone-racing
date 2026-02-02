#!/bin/bash
# Check training status on training PC
# Usage: ./training-status.sh

REPO="/home/ooousay/repos/isaac_drone_racer"

echo "=== Running Processes ==="
ssh training-pc 'wsl bash -c "ps aux | grep python | grep -v grep | grep -v unattended"' 2>/dev/null || echo "None"

echo ""
echo "=== Latest Training Runs ==="
ssh training-pc "wsl bash -c \"ls -lt $REPO/logs/skrl/drone_racer/ 2>/dev/null | head -6\"" || echo "No runs found"

echo ""
echo "=== Latest Checkpoints ==="
LATEST=$(ssh training-pc "wsl bash -c \"ls -t $REPO/logs/skrl/drone_racer/ 2>/dev/null | head -1\"")
if [ -n "$LATEST" ]; then
    ssh training-pc "wsl bash -c \"ls -la $REPO/logs/skrl/drone_racer/$LATEST/checkpoints/ 2>/dev/null | tail -5\"" || echo "No checkpoints"
fi

echo ""
echo "=== GPU Usage ==="
ssh training-pc 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader' 2>/dev/null || echo "Could not query GPU"
