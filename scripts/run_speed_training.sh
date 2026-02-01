#!/bin/bash
# Speed training - either finetune or train from scratch
# Usage: bash run_speed_training.sh [finetune|finetune2|train] [timesteps]

MODE=${1:-finetune}
TIMESTEPS=${2:-300000}

cd ~/repos/autonomous-drone-racing
source venv/bin/activate

if [ "$MODE" = "finetune" ]; then
    echo "Fine-tuning curriculum model for speed (conservative)..."
    python scripts/finetune_speed.py \
        --timesteps $TIMESTEPS \
        --envs 16 \
        --lr 1e-5 \
        --speed-bonus 0.5 \
        --lap-bonus 200 \
        2>&1 | tee finetune_speed.log
elif [ "$MODE" = "finetune2" ]; then
    echo "Fine-tuning curriculum model for speed (aggressive)..."
    python scripts/finetune_speed.py \
        --timesteps $TIMESTEPS \
        --envs 16 \
        --lr 3e-5 \
        --speed-bonus 2.0 \
        --lap-bonus 500 \
        2>&1 | tee finetune_speed2.log
else
    echo "Training speed from scratch..."
    python scripts/train_speed.py \
        --timesteps $TIMESTEPS \
        --envs 16 \
        --target-speed 5.0 \
        2>&1 | tee speed_train.log
fi
