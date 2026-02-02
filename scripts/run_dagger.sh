#!/bin/bash
set -e
cd ~/repos/autonomous-drone-racing
source ~/repos/pybullet-venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== DAGGER TRAINING ==="
echo "Started: $(date)"
echo "Hardware: RTX 5080, 64GB RAM, 24 cores"
echo ""

# Backup current best model
if [ -f models/vision_student/best_model.pt ]; then
    BACKUP="models/vision_student/best_model.pre_dagger.$(date +%Y%m%d_%H%M%S).pt"
    cp models/vision_student/best_model.pt "$BACKUP"
    echo "Backed up existing model to: $BACKUP"
fi

echo ""
echo "Running DAgger iterations..."
echo "- 3 iterations"
echo "- 100 episodes per iteration (more data coverage)"
echo "- 5 epochs per iteration (avoid overfitting)"
echo "- Beta: 0.5 → 0.25 → 0.125"
echo ""

python -u scripts/run_dagger.py \
    --student models/vision_student/best_model.pt \
    --teacher models/curriculum_final.zip \
    --data data/dart_demos \
    --output data/dagger \
    --iterations 3 \
    --episodes 100 \
    --epochs 5 \
    --beta-start 0.5 \
    --beta-decay 0.5 \
    --device cuda \
    2>&1 | tee /tmp/dagger.log

echo ""
echo "=== DAGGER COMPLETE ==="
echo "Finished: $(date)"

# Copy final model to main location
FINAL_MODEL=$(ls -t data/dagger/iter_*/model/best_model.pt 2>/dev/null | head -1)
if [ -n "$FINAL_MODEL" ]; then
    cp "$FINAL_MODEL" models/vision_student/best_model_dagger.pt
    echo "Final DAgger model: models/vision_student/best_model_dagger.pt"
fi

# Final evaluation
echo ""
echo "=== FINAL EVALUATION ==="
python -u scripts/eval_vision_student.py \
    --model models/vision_student/best_model_dagger.pt \
    --episodes 20 \
    --device cuda \
    2>&1 | tee /tmp/dagger_eval.log

echo ""
echo "=== ALL DONE ==="
