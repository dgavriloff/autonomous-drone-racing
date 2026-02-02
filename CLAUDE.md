# AI Grand Prix - Vision-Based Drone Racing

## Overview
Vision-only drone racing for Anduril competition. $500K prize.
- Qualification: April-June 2026
- Key: Camera-only perception (no ground truth state)

## Status: Path A (Swift-Aligned)

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Action Space Migration | IN PROGRESS | Thrust + body rates (not velocity) |
| 2. PPO Training | Ready | Swift hyperparameters implemented |
| 3. Gate Detector | Pending | GateNet exists, needs eval |
| 4. State Estimation | Pending | VIO + Kalman filter |
| 5. Sim-to-Real | Pending | Residual models |

## Architecture (Swift-Aligned)

```
Perception:
  Camera → GateNet → Gate corners → Pose estimation
  IMU → VIO → Kalman Filter → State estimate

Control:
  State (31-dim) → PPO Policy (128×128 MLP) → Thrust + Body Rates (4-dim)
```

### State Vector (31 dimensions)
- Position (3)
- Velocity (3)
- Rotation matrix flattened (9) - NOT quaternion
- Gate corners relative to drone (4×3 = 12)
- Previous action (4)

### Action Space
- Collective thrust (mass-normalized, m/s²)
- Roll rate, Pitch rate, Yaw rate (rad/s)

## Key Files

### New (Path A)
- `src/envs/swift_racing_env.py` - Swift-aligned environment
- `scripts/train_swift_ppo.py` - PPO training with Swift hyperparameters

### Existing
- `src/vision/gate_net.py` - Gate segmentation network
- `models/curriculum_final.zip` - Old SAC teacher (5/5 gates, velocity control)
- `USING_TRAINING_PC.md` - Remote training setup

## Training PC
See `USING_TRAINING_PC.md`. RTX 5080, 64GB RAM. **Use 16 parallel envs.**

## Commands

```bash
# Train Swift PPO (local test)
python scripts/train_swift_ppo.py --timesteps 100000 --envs 8

# Evaluate model
python scripts/train_swift_ppo.py --eval models/swift_ppo/final.zip

# Full training (on training PC)
python scripts/train_swift_ppo.py --timesteps 100000000 --envs 16
```

## Why Swift Architecture?

The original SAC + velocity approach fails at sim-to-real because:
1. Velocity control adds abstraction that breaks on real hardware
2. SAC is off-policy, less stable for precise control
3. No residual models to bridge reality gap

Swift won because their architecture was designed for sim-to-real:
- Thrust + body rates are closer to actuators
- PPO is on-policy, more stable
- Residual models bridge the reality gap
