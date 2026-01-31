# AI Grand Prix - Vision-Based Drone Racing

## Project Overview
Vision-based autonomous drone racing system for the AI Grand Prix competition by Anduril.

**Competition Timeline:**
- Qualification: April-June 2026
- Finals: November 2026
- Prize: $500K + job offer

**Key Requirement:** Camera-only perception (no ground truth state)

## Architecture

```
Camera (24Hz) -> GateNet -> QuAdGate -> EKF (500Hz) -> G&CNet -> Motors
```

### Components
- **GateNet**: U-Net segmentation for gate detection (src/vision/gate_net.py)
- **QuAdGate**: Corner detection from masks (src/vision/quad_gate.py)
- **PoseEstimator**: PnP-based gate pose (src/vision/pose_estimator.py)
- **EKF**: Extended Kalman Filter for state estimation (src/state/ekf.py)
- **G&CNet**: Neural network controller (src/control/gcnet.py)
- **MotorMixer**: Thrust/torque to RPM (src/control/motor_mixer.py)

## Quick Start

```bash
# Install dependencies
pip install -e .

# Collect training data
python scripts/train_gate_net.py --collect_data --num_frames 50000

# Train GateNet
python scripts/train_gate_net.py --epochs 10

# Train G&CNet (imitation learning)
python scripts/train_gcnet.py --phase imitation --collect_data

# Run pipeline
python scripts/run_pipeline.py --gui
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Control freq | 500 Hz | Target for competition |
| Vision freq | 24 Hz | Camera frame rate |
| Image size | 64x48 | gym-pybullet-drones default |
| GateNet params | ~400K | Lightweight for real-time |
| Max RPM | 65535 | Crazyflie 2.X |

## Testing Commands

```bash
# Test individual modules
python -m src.vision.gate_net
python -m src.vision.quad_gate
python -m src.state.ekf
python -m src.control.motor_mixer
python -m src.control.gcnet
python -m src.envs.high_freq_racing
python -m src.pipeline.vision_racing
```

## Current Performance
- PID Baseline: 4.97 m/s on monorace_11 (P_xy=1.05)
- 82.8% of MonoRace record (6.0 m/s)

## TODO
1. Train GateNet on collected data
2. Implement domain randomization for sim-to-real
3. Fine-tune G&CNet with PPO
4. Optimize for competition SDK
