#!/bin/bash
# Run a command in WSL on the training PC
# Usage: ./wsl-run.sh "command to run"
#        ./wsl-run.sh  (interactive shell)

set -e

if [ -z "$1" ]; then
    # Interactive shell
    ssh -t training-pc 'wsl'
else
    # Run command
    ssh training-pc "wsl bash -c \"$1\""
fi
