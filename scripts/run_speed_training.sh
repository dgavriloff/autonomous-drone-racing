#!/bin/bash
# Speed training - various modes
# Usage: bash run_speed_training.sh [curriculum|finetune|finetune2|train]

MODE=${1:-curriculum}

cd ~/repos/autonomous-drone-racing
source venv/bin/activate

if [ "$MODE" = "curriculum" ]; then
    echo "Full curriculum training with new SPEED_LIMIT..."
    python scripts/train_curriculum.py --envs 16 2>&1 | tee curriculum_fast.log
elif [ "$MODE" = "finetune" ]; then
    echo "Fine-tuning curriculum model for speed (conservative)..."
    python scripts/finetune_speed.py --timesteps 300000 --envs 16 --lr 1e-5 2>&1 | tee finetune_speed.log
elif [ "$MODE" = "finetune2" ]; then
    echo "Fine-tuning curriculum model for speed (aggressive)..."
    python scripts/finetune_speed.py --timesteps 300000 --envs 16 --lr 3e-5 --speed-bonus 2.0 2>&1 | tee finetune_speed2.log
else
    echo "Training speed from scratch..."
    python scripts/train_speed.py --timesteps 500000 --envs 16 2>&1 | tee speed_train.log
fi
