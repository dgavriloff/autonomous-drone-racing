#!/bin/bash
# Velocity Control Training with Dense Rewards
# Training PC: RTX 5080, 64GB RAM, 24 cores

set -e

cd /home/ooousay/repos/autonomous-drone-racing

# Activate venv
source /home/ooousay/repos/pybullet-venv/bin/activate

# Log file
LOGFILE=/home/ooousay/velocity_training.log

{
echo "=============================================="
echo "VELOCITY CONTROL TRAINING - DENSE REWARDS"
echo "=============================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Envs: 16 parallel"
echo "Timesteps: 100M"
echo "Started: $(date)"
echo ""
echo "Reward coefficients (FIXED):"
echo "  gate_bonus: 100.0"
echo "  progress: 50.0 per meter"
echo "  velocity: 2.0"
echo "  smoothness: -0.005"
echo "  crash: -20.0"
echo "=============================================="

# Run training with SAC (same as teacher model)
python scripts/train_parallel.py \
    --timesteps 100000000 \
    --envs 16 \
    --gates 5 \
    --radius 1.5 \
    --max-steps 1000 \
    --algorithm SAC

echo "=============================================="
echo "Training completed: $(date)"
echo "=============================================="
} 2>&1 | tee "$LOGFILE"
