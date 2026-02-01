# Aerial Gym Port Plan

## ⚠️ BLOCKED: RTX 5080 Blackwell Compatibility Issue

**Status**: Port BLOCKED - RTX 5080 (Blackwell sm_120) not supported by Isaac Sim 4.5.0/PyTorch

The training PC's RTX 5080 GPU is too new. PyTorch bundled with Isaac Sim 4.5.0 only supports up to Ada Lovelace (sm_90). Blackwell (sm_120) support requires Isaac Sim 5.2+ (expected Q1-Q2 2026).

**Options**:
1. Wait for Isaac Sim 5.2+ with Blackwell support
2. Swap training PC GPU to RTX 4090 or older
3. Continue optimizing gym-pybullet-drones (current approach)

---

## Overview
Port drone racing training from gym-pybullet-drones (PyBullet) to Aerial Gym (Isaac Sim) for 38,000x real-time performance and competition-ready speeds (30+ m/s).

## Why Port?
- **Speed ceiling**: CF2X maxes at 8.33 m/s, RACE drone has altitude drift issues
- **Performance**: Aerial Gym achieves 38,000x real-time vs ~1200 FPS with PyBullet
- **Competition relevance**: Better domain randomization for sim-to-real transfer
- **GPU acceleration**: Parallel environments on GPU, not CPU processes

## Requirements

### Hardware (Training PC)
- [x] NVIDIA RTX 5080 (confirmed)
- [x] CUDA support (confirmed)
- [ ] 50GB+ disk space for Isaac Sim
- [ ] NVIDIA driver 525.60+

### Software
- [ ] Ubuntu 22.04 on WSL2 (need to verify version)
- [ ] Isaac Sim 4.0+ (via Omniverse Launcher or pip)
- [ ] Aerial Gym from GitHub

## Phase 1: Environment Setup (Training PC)

### 1.1 Check Prerequisites
```bash
# Check WSL version and Ubuntu version
wsl --version
cat /etc/os-release

# Check NVIDIA driver version
nvidia-smi

# Check disk space
df -h ~
```

### 1.2 Install Isaac Sim
Option A: Omniverse Launcher (GUI needed - might not work in WSL)
Option B: pip install (headless, preferred for WSL)
```bash
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com
```

### 1.3 Install Aerial Gym
```bash
git clone https://github.com/ntnu-arl/aerial_gym_simulator.git
cd aerial_gym_simulator
pip install -e .
```

### 1.4 Verify Installation
```bash
python -c "from aerial_gym import AerialGym; print('Success')"
```

## Phase 2: Architecture Mapping

### 2.1 Environment Comparison

| gym-pybullet-drones | Aerial Gym |
|---------------------|------------|
| VelocityRacingEnv | AerialGym task |
| SubprocVecEnv (CPU) | IsaacGym vec env (GPU) |
| ActionType.VEL | Built-in velocity mode |
| 240Hz physics | Configurable (up to 2kHz) |
| PyBullet URDF | USD assets |

### 2.2 Key Files to Port

**Create new:**
- `src/aerial_gym/racing_task.py` - Racing environment task
- `src/aerial_gym/track.py` - Track/gate definitions
- `scripts/train_aerial_gym.py` - Training script

**Preserve logic from:**
- `scripts/train_parallel.py` - Reward shaping, observation space
- `scripts/train_curriculum.py` - Curriculum stages

### 2.3 Observation Space Mapping
```python
# Current (gym-pybullet-drones)
obs = [
    gate_vector (3),      # Relative position to next gate
    velocity (3),         # Current velocity
    orientation (4),      # Quaternion
    angular_velocity (3), # Angular rates
]  # Total: 13

# Aerial Gym (similar, may need adjustment)
obs = [
    root_positions,       # From env
    root_velocities,      # From env
    root_quats,          # From env
    gate_relative,       # Custom
]
```

### 2.4 Action Space
```python
# Current: [vx, vy, vz, yaw_rate] normalized [-1, 1]
# Aerial Gym: Supports velocity commands, thrust+torque, or direct RPM
# Keep velocity mode for consistency
```

## Phase 3: Implementation

### 3.1 Racing Task Class
```python
class DroneRacingTask(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Track setup
        self.gates = self._create_track(cfg.num_gates, cfg.radius)
        self.current_gate = torch.zeros(cfg.num_envs, device=sim_device)

        # Rewards (port from VelocityRacingEnv)
        self.gate_reward = 50.0
        self.progress_scale = 10.0
        self.velocity_penalty = 0.02

    def compute_observations(self):
        # Gate vector relative to drone
        # Velocity in body frame
        # Orientation

    def compute_reward(self):
        # Same reward structure as PyBullet version
```

### 3.2 Track Definition
```python
def create_simple_track(num_gates: int, radius: float) -> List[Gate]:
    """Same circular track as PyBullet version."""
    gates = []
    for i in range(num_gates):
        angle = 2 * np.pi * i / num_gates
        position = [radius * np.cos(angle), radius * np.sin(angle), 1.0]
        # Point gate toward center
        yaw = angle + np.pi
        gates.append(Gate(position, yaw))
    return gates
```

### 3.3 Training Script
```python
# Use rl_games or stable-baselines3 with Aerial Gym wrapper
from aerial_gym import AerialGymEnv
from stable_baselines3 import PPO

env = AerialGymEnv("DroneRacing", num_envs=4096)  # 4096 parallel on GPU!
model = PPO("MlpPolicy", env, device="cuda")
model.learn(total_timesteps=10_000_000)  # Fast with GPU parallelism
```

## Phase 4: Curriculum Port

### 4.1 Geometry Curriculum (same stages)
1. r=1.5m, 3 gates
2. r=1.5m, 5 gates
3. r=2.0m, 5 gates
4. r=2.5m, 5 gates

### 4.2 Speed Curriculum (expanded for higher speeds)
5. 5 m/s, r=2m
6. 10 m/s, r=3m
7. 15 m/s, r=5m
8. 20 m/s, r=7m
9. 25 m/s, r=10m
10. 30 m/s, r=15m (competition target!)

## Phase 5: Validation

### 5.1 Unit Tests
- [ ] Track creation matches PyBullet geometry
- [ ] Observation space produces valid outputs
- [ ] Reward computation matches expected values

### 5.2 Training Validation
- [ ] Model trains without crashes
- [ ] Reward increases over time
- [ ] Agent learns to pass gates

### 5.3 Performance Comparison
| Metric | PyBullet | Aerial Gym | Target |
|--------|----------|------------|--------|
| FPS | 1200 | 100,000+ | - |
| Max speed | 8.33 m/s | 30+ m/s | 30 m/s |
| Gates passed | 4-5/5 | 5/5 | 5/5 |

## Execution Timeline

### Day 1: Setup
- [ ] Verify training PC prerequisites
- [ ] Install Isaac Sim (pip method)
- [ ] Install Aerial Gym
- [ ] Run example script

### Day 2: Core Port
- [ ] Create racing_task.py
- [ ] Create track.py
- [ ] Port observation/reward logic

### Day 3: Training
- [ ] Create train_aerial_gym.py
- [ ] Run initial training (geometry phase)
- [ ] Debug and iterate

### Day 4+: Curriculum & Optimization
- [ ] Full curriculum training
- [ ] Speed optimization
- [ ] Domain randomization

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Isaac Sim doesn't work in WSL | Use native Windows or dual-boot Linux |
| API differences break logic | Study Aerial Gym examples thoroughly |
| GPU memory limits | Reduce num_envs if needed |
| Different physics | Tune rewards empirically |

## Commands Quick Reference

```bash
# SSH to training PC
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "COMMAND"'

# Check GPU
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "nvidia-smi"'

# Install Isaac Sim
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "pip install isaacsim-rl --extra-index-url https://pypi.nvidia.com"'

# Clone Aerial Gym
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "cd ~/repos && git clone https://github.com/ntnu-arl/aerial_gym_simulator.git"'
```

## References
- [Aerial Gym GitHub](https://github.com/ntnu-arl/aerial_gym_simulator)
- [Isaac Sim Docs](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [Aerial Gym Paper](https://arxiv.org/abs/2305.16510)
