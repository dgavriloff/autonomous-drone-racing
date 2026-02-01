#!/bin/bash
# Speed training with fine-tuning settings
cd ~/repos/autonomous-drone-racing
source ~/miniconda3/etc/profile.d/conda.sh
conda activate drone-racing
python scripts/train_speed.py \
    --timesteps 1000000 \
    --envs 24 \
    --target-speed 5.0 \
    --resume models/curriculum_final.zip \
    --lr 3e-5 \
    > speed.log 2>&1
