#!/bin/bash
# Speed training from scratch
cd ~/repos/autonomous-drone-racing
source venv/bin/activate
python scripts/train_speed.py \
    --timesteps 500000 \
    --envs 16 \
    --target-speed 5.0 \
    2>&1 | tee speed.log
