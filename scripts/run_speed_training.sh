#!/bin/bash
# Speed training - test different reward modes
# Usage: bash run_speed_training.sh [mode]
# Modes: default, min_speed, massive_lap

MODE=${1:-min_speed}

cd ~/repos/autonomous-drone-racing
source venv/bin/activate
echo "Starting training with mode: $MODE"
python scripts/train_speed.py \
    --timesteps 300000 \
    --envs 16 \
    --target-speed 5.0 \
    --mode $MODE \
    2>&1 | tee speed_${MODE}.log
