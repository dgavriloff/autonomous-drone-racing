# AI Grand Prix - Vision-Based Drone Racing

## Project Overview
Vision-based autonomous drone racing system for the AI Grand Prix competition by Anduril.

**Competition Timeline:**
- Qualification: April-June 2026
- Finals: November 2026
- Prize: $500K + job offer

**Key Requirement:** Camera-only perception (no ground truth state)

## Current Status (Updated 2026-02-02)

| Component | Status | Performance |
|-----------|--------|-------------|
| Teacher Policy | ✅ Done | 5/5 gates (ground truth) |
| Vision Student (BC) | ✅ Done | 3.2/5 gates |
| DAgger | ✅ Tried | 3.2/5 gates (marginal improvement) |
| **RL Fine-tuning** | 🔄 **NEXT** | Target: 5/5 gates |

### Current Bottleneck

**Diagnosis found:** Gates 1-3 pass 100%, Gates 4-5 fail 80%.
- Vision works (model CAN see gates)
- Problem is cumulative error / precision at later gates
- RL should help (optimizes for task completion, not action matching)

## IMPORTANT: Training PC Guidelines

See `USING_TRAINING_PC.md` for full instructions.

**Key lessons learned:**
- Use shell scripts with tmux, NOT inline commands
- BC loss does NOT correlate with task performance
- Save checkpoints every epoch, eval on actual gates passed
- Use 16 parallel envs (SubprocVecEnv) to utilize hardware

## Next Step: RL Fine-Tuning

**Plan (from Entry 48 in blogs.md):**

1. Verify state-based RL works (teacher already does 5/5)
2. Create vision RL env (camera obs, velocity actions)
3. Initialize PPO from BC checkpoint (start at 3/5)
4. Fine-tune with gate rewards + BC regularization
5. Use 16 parallel envs (finally use the RTX 5080 properly!)

**Expected outcome:** 3.2/5 → 5/5 gates

## Key Files

### Models
- `models/curriculum_final.zip` - Teacher policy (5/5 gates, state-based)
- `models/vision_student/best_model.pt` - BC student (3.2/5 gates, vision)
- `data/dagger/iter_02/model/best_model.pt` - Best DAgger model

### Scripts
- `scripts/train_parallel.py` - State-based RL training
- `scripts/train_vision_student.py` - Behavioral cloning
- `scripts/run_dagger.py` - DAgger pipeline
- `scripts/diagnose_student.py` - Diagnose where model fails
- `scripts/eval_vision_student.py` - Evaluate vision model

### Data
- `data/dart_demos/` - 94K DART demos (noise-injected)
- `data/dagger/` - DAgger aggregated data (~400K frames)

## Architecture

### Vision Pipeline (Working)
```
Camera (64x48 RGB, 4 frames)
    ↓
VisionStudentNetV2 (CNN encoder + MLP, 1.2M params)
    ↓
Velocity commands [vx, vy, vz, yaw_rate]
```

### What's Been Tried

| Approach | Result | Issue |
|----------|--------|-------|
| Pure BC | 2.9/5 | Compounding error |
| DART + Frame stacking | 3.0/5 | Still compounding |
| DAgger (3 iterations) | 3.2/5 | Underfitting, slow collection |
| **RL Fine-tuning** | TBD | **Next step** |

## Training PC

RTX 5080, 64GB RAM, 24 cores via SSH + Tailscale.

```bash
# Sync and run
git push
ssh ooousay@denis.tail07d7b1.ts.net "wsl git -C /home/ooousay/repos/autonomous-drone-racing pull"
ssh ooousay@denis.tail07d7b1.ts.net 'wsl tmux new-session -d -s train /path/to/script.sh'
```

## TODO

1. ✅ ~~Train teacher policy~~ (5/5 gates)
2. ✅ ~~Train BC student~~ (3.2/5 gates)
3. ✅ ~~Try DAgger~~ (marginal improvement)
4. ✅ ~~Diagnose failures~~ (gates 4-5 are the problem)
5. 🔄 **RL Fine-tuning** ← CURRENT
6. ⬜ Multi-track generalization
7. ⬜ Domain randomization
8. ⬜ DCL SDK integration (April 2026)
