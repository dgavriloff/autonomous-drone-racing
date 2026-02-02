# AI Grand Prix - Vision-Based Drone Racing

## Project Overview
Vision-based autonomous drone racing system for the AI Grand Prix competition by Anduril.

**Competition Timeline:**
- Qualification: April-June 2026
- Finals: November 2026
- Prize: $500K + job offer

**Key Requirement:** Camera-only perception (no ground truth state)

## Current Status (Updated 2026-02-02)

| Component | Status | Details |
|-----------|--------|---------|
| Teacher Policy | ✅ Done | `models/curriculum_final.zip` - 5/5 gates |
| GateNet | ✅ Done | `models/gate_net/best_model.pt` - 76% IoU |
| Vision Demos | ✅ Done | 44K frames in `data/teacher_demos/` |
| Vision Student | 🔄 Next | Behavioral cloning ready to train |

## Simulator: gym-pybullet-drones ONLY

**DO NOT use Isaac Sim, Flightmare, or AirSim.** We evaluated all options:

| Simulator | Status | Why Not |
|-----------|--------|---------|
| **gym-pybullet-drones** | ✅ USE THIS | Works for everything |
| Isaac Sim | ❌ Abandoned | Vulkan broken in WSL2, can't render cameras |
| Flightmare | ❌ Dead | Python 3.6, TensorFlow 1.14, unmaintained since 2020 |
| AirSim | ❌ Deprecated | Microsoft killed it Dec 2023 |

## Training PC

**Use for long training runs.** RTX 5080, 64GB RAM, 24 cores.

```bash
# SSH to training PC
ssh ooousay@denis.tail07d7b1.ts.net

# Run command in WSL
wsl <command>

# Example: pull and train
wsl bash -c "cd ~/repos/autonomous-drone-racing && git pull"
```

Note: No venv on training PC for pybullet. Install deps as needed or create one.

## Architecture

### Current: Teacher-Student Pipeline
```
1. Teacher (ground truth) → curriculum_final.zip [DONE]
2. Collect demos (image, action) → 44K frames [DONE]
3. Train vision student (behavioral cloning) → [IN PROGRESS]
4. Test vision-only flight → [NEXT]
```

### Vision Pipeline
```
Camera (64x48 RGB)
    ↓
GateNet (U-Net, 482K params) → Binary mask
    ↓
QuAdGate → 4 corner points
    ↓
PoseEstimator (PnP) → Gate pose
    ↓
Policy → Velocity commands [vx, vy, vz, yaw_rate]
```

## Key Files

### Models
- `models/curriculum_final.zip` - Teacher policy (5/5 gates)
- `models/gate_net/best_model.pt` - GateNet segmentation (76% IoU)

### Scripts
- `scripts/train_parallel.py` - Train state-based policy
- `scripts/collect_teacher_demos.py` - Collect vision demos from teacher
- `scripts/train_vision_student.py` - Behavioral cloning training
- `scripts/test_vision_pipeline.py` - Test full vision pipeline

### Data
- `data/teacher_demos/` - 44K (image, action) pairs for BC training

## Quick Start

```bash
# Activate environment (Mac)
conda activate drone-racing

# Train vision student
python scripts/train_vision_student.py --demos data/teacher_demos --epochs 100

# Test vision pipeline
python scripts/test_vision_pipeline.py --model models/curriculum_final.zip
```

## Speed Limits

| Config | Speed | Notes |
|--------|-------|-------|
| Default | 0.25 m/s | Library hardcoded (SPEED_LIMIT bug) |
| Fixed | 4.17 m/s | Override in train_parallel.py |
| CF2X Max | 8.33 m/s | Drone physics limit |

Speed is NOT the bottleneck. Vision is. 8 m/s is plenty for validation.

## Archived (Do Not Use)

The following files are historical and relate to abandoned Isaac Sim work:
- `archive/AERIAL_GYM_PORT_PLAN.md`
- `archive/COMPETITION_DRONE_SPECS.md`
- `archive/IMPROVEMENT_PLAN.md`
- `archive/OPTIMIZATION_LOOP.md`

## TODO

1. ✅ ~~Train teacher policy~~ (5/5 gates)
2. ✅ ~~Train GateNet~~ (76% IoU)
3. ✅ ~~Collect vision demos~~ (44K frames)
4. 🔄 **Train vision student** ← CURRENT
5. ⬜ Test vision-only flight
6. ⬜ DAgger refinement (if needed)
7. ⬜ Domain randomization
8. ⬜ DCL SDK integration (April 2026)
