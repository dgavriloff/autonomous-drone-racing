# AI Grand Prix - Vision-Based Drone Racing

## Project Overview
Vision-based autonomous drone racing system for the AI Grand Prix competition by Anduril.

**Competition Timeline:**
- Qualification: April-June 2026
- Finals: November 2026
- Prize: $500K + job offer

**Key Requirement:** Camera-only perception (no ground truth state)

## IMPORTANT: Use Training PC for Training!

**Always use the remote training PC for training runs.** See `USING_TRAINING_PC.md` for details.
- RTX 5080, 64GB RAM, 24 cores
- ~1200 FPS with 16 parallel envs
- Push code via git, pull on remote, run with tmux

```bash
# Push code first
git push

# Pull and train on remote
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "cd ~/repos/autonomous-drone-racing && git pull && source venv/bin/activate && python scripts/YOUR_SCRIPT.py"'
```

## Current Status

| Metric | Value |
|--------|-------|
| Best gates | **5/5** |
| Approach | Velocity control (ActionType.VEL) |
| Best model | `models/curriculum_final.zip` |
| Training | Parallel envs (16x SubprocVecEnv) |
| Speed | 0.25 m/s (needs improvement - competition is 25+ m/s) |

## Architecture

### Current Working Approach (Velocity Control)
```
Observation -> SAC Policy -> Velocity Commands -> PID (internal) -> Motors
```
- Uses `ActionType.VEL` which provides velocity abstraction
- Built-in PID handles motor coordination
- Action `[vx, vy, vz, yaw_rate]` maps directly to intended movement

### Target Competition Architecture (Vision-Based)
```
Camera (24Hz) -> GateNet -> QuAdGate -> EKF (500Hz) -> G&CNet -> Motors
```

### Components
- **GateNet**: U-Net segmentation for gate detection (src/vision/gate_net.py)
- **QuAdGate**: Corner detection from masks (src/vision/quad_gate.py)
- **PoseEstimator**: PnP-based gate pose (src/vision/pose_estimator.py)
- **EKF**: Extended Kalman Filter for state estimation (src/state/ekf.py)
- **G&CNet**: Neural network controller (src/control/gcnet.py) - NOT WORKING with direct RPM
- **MotorMixer**: Thrust/torque to RPM (src/control/motor_mixer.py)

## Project Files

### Key Documentation
- **`CLAUDE.md`** (this file): Project overview and instructions
- **`USING_TRAINING_PC.md`**: How to use the remote training PC via SSH/Tailscale. Use this for long training runs - the PC has an RTX 5080, 64GB RAM, and 24 cores. Always sync code via git before training.
- **`blogs.md`**: Development blog documenting discoveries, bugs, and solutions. **Add entries here when making significant progress or discoveries.** Follow the existing format with date, problem, solution, and lessons learned.

### Training Scripts
- **`scripts/train_parallel.py`**: Main training script with parallel envs (16x SubprocVecEnv)
- **`scripts/test_parallel_model.py`**: Test trained models with configurable tolerance/steps

## Environment Setup

**Conda environment**: `drone-racing` (miniconda3)

```bash
# Activate environment
source /Users/denisgavriloff/miniconda3/etc/profile.d/conda.sh
conda activate drone-racing
```

## Quick Start

```bash
# Train with parallel environments (local)
python scripts/train_parallel.py --timesteps 1000000 --envs 16

# Test the model
python scripts/test_parallel_model.py --model models/parallel_vel/large_tolerance.zip --episodes 5

# Test with GUI visualization
python scripts/test_parallel_model.py --model models/parallel_vel/large_tolerance.zip --episodes 3
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Control freq | 48 Hz | With velocity abstraction |
| Physics freq | 240 Hz | PyBullet simulation |
| Gate tolerance | 0.8m (train) / 1.0m (test) | Larger helps with altitude drift |
| Max steps | 500 (train) / 1000 (test) | More time to reach later gates |
| Parallel envs | 16 | ~1200 FPS on training PC |
| Max RPM | 21702.64 | CF2X actual (NOT 65535) |

## Important Lessons Learned

### 1. Action Space Matters More Than Reward
Direct motor control (ActionType.RPM) is extremely hard to learn. The agent kept flying straight up regardless of reward shaping. **Velocity control (ActionType.VEL) was the breakthrough** - it provides intuitive `[vx, vy, vz, yaw_rate]` commands while the built-in PID handles motor coordination.

### 2. Frequency Matching
Expert data collection MUST use the same control frequency as inference. Collecting at 240Hz and running at 500Hz causes crashes because physics timestep affects required motor thrust.

### 3. MAX_RPM Constant
gym-pybullet-drones CF2X has MAX_RPM ~21702, NOT 65535. Using wrong value causes 48% thrust loss.

### 4. Research Recommendations Don't Always Transfer
Literature suggested: fixed entropy coefficient, relative observations, VecNormalize. **All made things worse for our task.** Always test empirically.

### 5. Altitude Drift is Real
The drone navigates horizontally but drifts vertically (~0.7m). Solution: larger gate tolerance (0.8m training, 1.0m testing).

### 6. Analyze Trajectories, Not Just Metrics
Reward curves looked fine, but watching actual position over time revealed the altitude drift problem.

## Current Challenges

### Speed is Too Slow
- Current: 0.25 m/s average
- Competition: 25+ m/s (100x faster!)
- Need speed-optimized training with lap time rewards

### Vision Pipeline Blocked
- DCL SDK not yet released (coming April 2026)
- GateNet trained on wrong data (PyBullet ≠ DCL visuals)
- Architecture is ready, just needs DCL training data

### Policy Robustness (SOLVED)
- Policy handles 10cm noise + 40ms delay
- Ready for vision integration when DCL data available

## TODO
1. ~~Train GateNet on collected data~~
2. ~~Get velocity control working~~ (5/5 gates ✓)
3. ~~Get 5/5 gates~~ (curriculum learning ✓)
4. **Speed optimization** (current priority - 0.25 m/s → 10+ m/s)
5. Harder tracks (more gates, altitude variation)
6. Domain randomization for sim-to-real
7. Vision integration (blocked until DCL SDK)
