#!/bin/bash
# Kill all training/eval processes on training PC
# Usage: ./kill-training.sh

echo "Finding python processes..."
PIDS=$(ssh training-pc 'wsl bash -c "ps aux | grep python | grep -v grep | grep -v unattended | awk \"{print \\\$2}\""' 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "No python processes found"
    exit 0
fi

echo "Found PIDs: $PIDS"
echo "Killing..."

for pid in $PIDS; do
    ssh training-pc "wsl bash -c \"kill -9 $pid 2>/dev/null\"" && echo "Killed $pid" || echo "Failed to kill $pid"
done

echo "Done"
