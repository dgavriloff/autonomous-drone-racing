#!/bin/bash
# Resume training from best checkpoint (1.6M steps, ~4.29 avg gates)

set -e

cd /home/ooousay/repos/autonomous-drone-racing

# Activate venv
source /home/ooousay/repos/pybullet-venv/bin/activate

# Log file
LOGFILE=/home/ooousay/velocity_training_resumed.log

{
echo "=============================================="
echo "RESUMING FROM 1.6M CHECKPOINT"
echo "=============================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Checkpoint: parallel_1600000_steps.zip"
echo "Started: $(date)"
echo "=============================================="

python scripts/train_parallel.py \
    --timesteps 100000000 \
    --envs 16 \
    --gates 5 \
    --radius 1.5 \
    --max-steps 1000 \
    --algorithm SAC \
    --resume models/parallel_vel/parallel_1600000_steps.zip

echo "=============================================="
echo "Training completed: $(date)"
echo "=============================================="
} 2>&1 | tee "$LOGFILE"
