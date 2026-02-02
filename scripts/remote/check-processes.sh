#!/bin/bash
# Check running python processes on training PC
# Usage: ./check-processes.sh

ssh training-pc 'wsl bash -c "ps aux | grep python | grep -v grep | grep -v unattended"' 2>/dev/null || echo "No python processes running"
