#!/bin/bash
# Start Isaac Drone Racer training on the training PC
# Usage: ./start-training.sh [iterations] [num_envs]
#        ./start-training.sh 100000 4096

ITERATIONS=${1:-50000}
NUM_ENVS=${2:-4096}
REPO="/home/ooousay/repos/isaac_drone_racer"
VENV="/home/ooousay/repos/isaac-racing-venv"

echo "Starting training: $ITERATIONS iterations, $NUM_ENVS envs"

# Use tmux to keep training alive after SSH disconnect
ssh training-pc "wsl bash -c \"tmux kill-session -t training 2>/dev/null; tmux new-session -d -s training 'cd $REPO && source $VENV/bin/activate && python scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --num_envs $NUM_ENVS --max_iterations $ITERATIONS 2>&1 | tee training.log'\""

echo "Training started in tmux session 'training'"
echo ""
echo "To monitor: ./monitor-training.sh"
echo "To attach:  ssh -t training-pc 'wsl bash -c \"tmux attach -t training\"'"
