#!/bin/bash
# Swift PPO Training Script for Training PC
# RTX 5080, 64GB RAM, 24 cores

set -e

cd /home/ooousay/repos/autonomous-drone-racing

# Activate venv
source /home/ooousay/repos/pybullet-venv/bin/activate

echo "=============================================="
echo "SWIFT PPO TRAINING"
echo "=============================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Envs: 16 parallel"
echo "Timesteps: 100M"
echo "Started: $(date)"
echo "=============================================="

# Run training with Swift hyperparameters
# 100M timesteps, 16 parallel envs, checkpoint every 500k
python scripts/train_swift_ppo.py \
    --timesteps 100000000 \
    --envs 16 \
    --gates 5 \
    --radius 1.5 \
    --max-steps 1000

echo "=============================================="
echo "Training completed: $(date)"
echo "=============================================="
