# AI Grand Prix - Vision-Based Drone Racing

## Overview
Vision-only drone racing for Anduril competition. $500K prize.
- Qualification: April-June 2026
- Key: Camera-only perception (no ground truth state)

## Status

| Component | Performance |
|-----------|-------------|
| Teacher (ground truth) | 5/5 gates |
| Vision Student (BC) | 3.2/5 gates |
| **Next: RL Fine-tuning** | Target 5/5 |

**Bottleneck:** Gates 1-3 pass 100%, gates 4-5 fail. Cumulative error, not vision.

## Key Files
- `models/curriculum_final.zip` - Teacher policy
- `models/vision_student/best_model.pt` - Best vision model
- `scripts/train_parallel.py` - RL training
- `scripts/eval_vision_student.py` - Evaluate vision model
- `blogs.md` - Development history + detailed plans

## Training PC
See `USING_TRAINING_PC.md`. RTX 5080, 64GB RAM. **Use 16 parallel envs.**

## Architecture
```
Camera (64x48 RGB, 4 frames) → VisionStudentNetV2 (1.2M params) → Velocity commands
```
