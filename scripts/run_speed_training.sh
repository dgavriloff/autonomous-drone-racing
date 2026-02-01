#!/bin/bash
# Speed training - either finetune or train from scratch
# Usage: bash run_speed_training.sh [finetune|train] [timesteps]

MODE=${1:-finetune}
TIMESTEPS=${2:-300000}

cd ~/repos/autonomous-drone-racing
source venv/bin/activate

if [ "$MODE" = "finetune" ]; then
    echo "Fine-tuning curriculum model for speed..."
    python scripts/finetune_speed.py \
        --timesteps $TIMESTEPS \
        --envs 16 \
        --lr 1e-5 \
        2>&1 | tee finetune_speed.log
else
    echo "Training speed from scratch..."
    python scripts/train_speed.py \
        --timesteps $TIMESTEPS \
        --envs 16 \
        --target-speed 5.0 \
        2>&1 | tee speed_train.log
fi
