#!/bin/bash
# Swift PPO Training Script for Training PC
# RTX 5080, 64GB RAM, 24 cores

set -e

cd /home/ooousay/repos/autonomous-drone-racing

# Activate venv
source /home/ooousay/repos/pybullet-venv/bin/activate

# Log file
LOGFILE=/home/ooousay/swift_training.log

{
echo "=============================================="
echo "SWIFT PPO TRAINING"
echo "=============================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Envs: 16 parallel"
echo "Timesteps: 100M"
echo "Started: $(date)"
echo "=============================================="

# Run training with Swift hyperparameters
python scripts/train_swift_ppo.py \
    --timesteps 100000000 \
    --envs 16 \
    --gates 5 \
    --radius 1.5 \
    --max-steps 1000

echo "=============================================="
echo "Training completed: $(date)"
echo "=============================================="
} 2>&1 | tee "$LOGFILE"
