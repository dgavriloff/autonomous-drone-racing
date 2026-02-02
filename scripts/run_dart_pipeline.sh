#!/bin/bash
cd ~/repos/autonomous-drone-racing
source ~/repos/pybullet-venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== DART DEMO COLLECTION ==="
python -u scripts/collect_teacher_demos.py --teacher models/curriculum_final.zip --episodes 100 --output data/dart_demos --noise 0.15 --no-features 2>&1 | tee /tmp/dart.log

echo "=== TRAINING WITH FRAME STACKING ==="
python -u scripts/train_vision_student.py --demos data/dart_demos --epochs 100 --batch-size 2048 --num-frames 4 --device cuda 2>&1 | tee /tmp/train_dart.log

echo "=== EVALUATION ==="
python -u scripts/eval_vision_student.py --model models/vision_student/best_model.pt --episodes 10 --device cuda 2>&1 | tee /tmp/eval_dart.log

echo "=== COMPLETE ==="
