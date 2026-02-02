#!/bin/bash
# Start Isaac Drone Racer training on the training PC
# Usage: ./start-training.sh [iterations] [num_envs]
#        ./start-training.sh 100000 4096

ITERATIONS=${1:-50000}
NUM_ENVS=${2:-4096}
REPO="/home/ooousay/repos/isaac_drone_racer"
VENV="/home/ooousay/repos/isaac-racing-venv"

echo "Starting training: $ITERATIONS iterations, $NUM_ENVS envs"

# Create training script on remote (avoids quoting hell)
# Note: LD_LIBRARY_PATH fix required for Isaac Sim GPU physics in WSL2
echo "#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:\$LD_LIBRARY_PATH
cd $REPO
source $VENV/bin/activate
python scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --num_envs $NUM_ENVS --max_iterations $ITERATIONS 2>&1 | tee training.log" | ssh training-pc 'wsl bash -c "cat > /tmp/run_training.sh && chmod +x /tmp/run_training.sh"'

# Kill any existing training session and start new one
ssh training-pc 'wsl bash -c "tmux kill-session -t training 2>/dev/null; tmux new-session -d -s training /tmp/run_training.sh"'

echo "Training started in tmux session 'training'"
echo ""
echo "To monitor: ./monitor-training.sh"
echo "To attach:  ssh -t training-pc 'wsl bash -c \"tmux attach -t training\"'"
