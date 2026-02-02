#!/bin/bash
set -e
cd ~/repos/autonomous-drone-racing
source ~/repos/pybullet-venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== RL FINE-TUNING ==="
echo "Started: $(date)"
echo "Hardware: RTX 5080, 64GB RAM, 24 cores"
echo "Using 16 parallel envs for efficiency"
echo ""

python -u scripts/rl_finetune_vision.py \
    --timesteps 500000 \
    --envs 16 \
    --eval-freq 50000 \
    2>&1 | tee /tmp/rl_finetune.log

echo ""
echo "=== RL FINE-TUNING COMPLETE ==="
echo "Finished: $(date)"
