#!/bin/bash
set -e  # Exit on error
cd ~/repos/autonomous-drone-racing
source ~/repos/pybullet-venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== VISION STUDENT TRAINING ==="
echo "Started: $(date)"
echo ""

# Backup existing model
if [ -f models/vision_student/best_model.pt ]; then
    BACKUP="models/vision_student/best_model.$(date +%Y%m%d_%H%M%S).pt"
    cp models/vision_student/best_model.pt "$BACKUP"
    echo "Backed up existing model to: $BACKUP"
fi

# Run training
echo ""
echo "Starting training..."
python -u scripts/train_vision_student.py \
    --demos data/dart_demos \
    --epochs 100 \
    --batch-size 2048 \
    --num-frames 4 \
    --device cuda \
    2>&1 | tee /tmp/training.log

echo ""
echo "=== TRAINING COMPLETE ==="
echo "Finished: $(date)"

# Run eval
echo ""
echo "=== RUNNING EVALUATION ==="
python -u scripts/eval_vision_student.py \
    --model models/vision_student/best_model.pt \
    --episodes 10 \
    --device cuda \
    2>&1 | tee /tmp/eval.log

echo ""
echo "=== ALL DONE ==="
