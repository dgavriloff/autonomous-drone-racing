# AI Grand Prix Development Blog

## Entry 1: The Frequency Mismatch Discovery
**Date: 2026-01-31**

### Where We Are
We've built out the full vision-based racing architecture for the AI Grand Prix competition ($500K prize!). The pipeline includes:
- GateNet (U-Net segmentation) - trained to IoU of 1.0
- QuAdGate (corner detection from masks)
- EKF for state estimation
- G&CNet (neural network controller) - trained to validation loss of 0.000002

### The Problem
The drone crashes after ~62 steps (0.12 seconds). After debugging, we discovered the **root cause: frequency mismatch**.

Expert data was collected using `CtrlAviary` which defaults to 240Hz control frequency. However, our `HighFreqRacingAviary` runs at 500Hz. The physics behave differently at different timesteps!

**Evidence:**
```
Thrust x1.0: Crashed at step 59, height=0.050
Thrust x3.0: Survived 200 steps, height=0.103
```

The learned hover RPMs (14476) are insufficient at 500Hz - they need ~3x the thrust to maintain altitude because the physics simulation integrates over smaller timesteps.

### The Fix
Re-collecting expert data at 500Hz using `HighFreqRacingAviary` instead of `CtrlAviary`. This ensures the neural network learns the correct motor commands for the actual inference frequency.

### Lessons Learned
1. **Always match training and inference physics** - control frequency matters!
2. **Start simple** - bypassing EKF with ground truth helped isolate the issue
3. **Test assumptions** - the 3x thrust test proved the frequency hypothesis

---

## Entry 2: The MAX_RPM Bug
**Date: 2026-01-31**

### The Frequency Match Was Just the Start

Re-collected data at 500Hz, but the drone still crashed after ~65 steps. Debugging revealed **the real issue: MAX_RPM mismatch**.

gym-pybullet-drones calculates MAX_RPM from drone physics (~21,702 for CF2X), but our code assumed 65,535. This meant:
- Model outputs 21,000 RPM (near max)
- Pipeline normalizes: 21000/65535 = 0.32
- Env applies: 0.32 × 21702 = 6945 RPM
- **Actual hover needs: 14,468 RPM**

The drone was getting ~48% of the thrust it needed!

### The Fix
1. Updated `PipelineConfig.max_rpm` to 21702.64
2. Updated `GCNet` default max_rpm to 21702.64
3. Fixed `create_gcnet` to load max_rpm from checkpoint config
4. Fixed `run_pipeline.py` to use correct model paths

### Results After Fix
| Metric | Before | After |
|--------|--------|-------|
| Steps | 64 | 875 |
| Reward | -102 | +307 |
| Avg Speed | 0.15 m/s | 5.30 m/s |
| Max Speed | 0.58 m/s | 9.97 m/s |

The drone is now flying fast! Still no gates passed though - need to debug navigation.

---

## Entry 3: Reward Hacking
**Date: 2026-01-31**

### PPO Learns to Fly... Straight Up

After fixing the MAX_RPM bug, tried PPO RL training. Results:
- Reward: 930 (great!)
- Episode length: 1000 (max - no crashes!)
- Gates passed: 0 (uh oh)

Traced the drone trajectory and found it flies **straight up at 8.8 m/s** and never moves toward gates. Classic reward hacking!

### The Problem

Original reward:
- `velocity_bonus: +0.1 * speed` (continuous)
- `gate_passed: +10` (one-time)
- `time_penalty: -0.01` (continuous)

Flying fast without navigating gives ~8.8 * 0.1 = +0.88/step.
Passing gates is risky and only gives +10 once.

### The Fix

Updated reward function:
- `velocity_bonus: +0.01 * speed` (reduced 10x)
- `gate_passed: +100` (increased 10x)
- `progress_bonus: +1.0 * (prev_dist - curr_dist)` (new - reward for approaching gates)

### Current Status

Retraining PPO with new reward. Will see if the drone learns to navigate toward gates instead of just flying up.

---

## Entry 4: Still Flying Up
**Date: 2026-01-31**

### V2 Results

Trained with new reward function. Results:
- Reward: 84 (down from 930)
- Gates passed: Still 0

Checked trajectory - **still flying straight up at 9.6 m/s!**

The progress reward gets triggered (distance to gate briefly decreases from 3.13m to 3.00m when flying up past gate height) but then gets worse. The agent found a local optimum.

### What's Working
1. Drone flies without crashing (1000 steps = full episode)
2. Model learns stable flight
3. Velocity is impressive (9.6 m/s)

### What's Not Working
1. Zero horizontal navigation
2. Ignores gate targets in observation
3. Reward shaping still allows reward hacking

### Next Steps

Options:
1. Much stronger progress reward (10x?)
2. Add negative reward for being far from gates
3. Curriculum learning (start closer to gates)
4. Different algorithm (SAC, TD3?)
5. Behavior cloning from a working trajectory + RL fine-tuning

### Current Status

The architecture is solid:
- GateNet: Trained (IoU ~1.0)
- G&CNet: Architecture works
- Pipeline: Functional
- Environment: 500Hz control working

The challenge is teaching navigation through RL. The agent keeps finding ways to maximize reward without actually racing through gates. This is a classic exploration/exploitation problem.

---

## Summary So Far

### Key Bugs Fixed
1. **Frequency mismatch**: Training at 240Hz, running at 500Hz
2. **MAX_RPM mismatch**: Using 65535 instead of 21702
3. **Model path defaults**: Pipeline loaded untrained model
4. **create_gcnet**: Didn't use checkpoint config

### Key Learnings
1. Always match training and inference frequencies
2. Verify all numerical constants from the actual library
3. Watch for reward hacking in RL
4. Reward shaping is hard - easy to create unintended optima

### Performance Progress
| Version | Steps | Reward | Speed | Gates |
|---------|-------|--------|-------|-------|
| Initial | 64 | -102 | 0.15 m/s | 0 |
| MAX_RPM fix | 875 | +307 | 5.30 m/s | 0 |
| PPO v1 | 1000 | +930 | 8.8 m/s | 0 |
| PPO v2 | 1000 | +84 | 9.6 m/s | 0 |

The drone flies great, but still no gates passed. More work needed on navigation.

---

## Entry 5: Parallel Research & PBRS Solution
**Date: 2026-01-31**

### Research Phase

Launched 4 parallel subagents to explore different approaches:

| Approach | Result |
|----------|--------|
| **Aggressive PID Tuning** | 4.97 m/s (1.3% improvement), stable |
| **RL (PPO)** | 5.41 m/s but only 2/11 gates, crashes |
| **CasADi Trajectory** | 4.94 m/s, theoretical optimum untrackable |
| **Racing Lines** | 4.92 m/s, baseline is already optimal |

### Key Insight: PBRS (Potential-Based Reward Shaping)

The RL agent keeps flying up because:
- Old reward: `+0.01 * speed` rewards ANY velocity
- Flying up at 9.6 m/s = steady +0.096/step
- Gate bonus (+100) is one-time and risky

**Solution: PBRS with directional velocity**

PBRS formula: `r_shaped = r + γ×Φ(s') - Φ(s)` where:
- `Φ(s) = -distance_to_gate` (closer = higher potential)
- `γ = 0.99` (discount factor)

Key changes:
1. **PBRS shaping**: Guides towards gates without changing optimal policy
2. **Distance penalty**: `-0.5 * dist_to_gate` prevents flying away
3. **Directional velocity**: Only reward velocity TOWARDS gate, not any direction

### Recommended Approach

Based on research, the best path forward:
1. **Use SAC** instead of PPO (better exploration for local optima)
2. **PBRS reward** (implemented in high_freq_racing.py)
3. **Curriculum learning** if still stuck (start gates closer)

### Current Status

- PBRS reward function committed
- Need to create SAC training script
- Need to test if drone now navigates to gates

---

## Entry 6: The Breakthrough - Velocity Control
**Date: 2026-01-31**

### The Real Problem: Raw Motor RPMs Are Too Hard to Learn

After trying SAC with PBRS, curriculum learning with progressively harder stages, and position-delta actions, nothing worked. The agent kept getting stuck or flying vertically.

**Root cause discovered through research:**
- `ActionType.RPM` (raw motor commands) requires learning complex motor coordination
- Hover action [0.5, 0.5, 0.5, 0.5] produces virtually NO movement
- Agent needs to learn: motor 1 vs motor 3 differential → pitch → forward movement
- This is too hard to learn from scratch!

### The Solution: ActionType.VEL (Velocity Control)

Based on extensive research of gym-pybullet-drones documentation and successful drone racing papers:

1. **gym-pybullet-drones has ActionType.VEL** which provides velocity commands
2. **Built-in PID controller** handles motor coordination
3. Action `[1, 0, 0, 0]` = move forward (intuitive!)
4. This is similar to how Swift (Nature paper) uses thrust + body rates

### Implementation

Created `train_velocity_control.py`:
- Uses `ActionType.VEL` instead of `ActionType.RPM`
- PBRS reward for progress towards gates
- Velocity alignment bonus
- Action smoothness penalty

### Results

**Training (200K steps, ~17 minutes):**
- Best: 2 gates
- Peak avg: 1.80 gates at 110K steps

**Testing (10 episodes):**
| Metric | Value |
|--------|-------|
| Gates passed | **2/5 consistently** |
| Horizontal travel | **2.05m** |
| Vertical travel | **0.15m** |
| Navigation | **HORIZONTAL** |

### Key Comparison

| Approach | Action Type | Gates | Behavior |
|----------|-------------|-------|----------|
| PPO v1 | RPM | 0 | Flies straight up |
| PPO v2 (PBRS) | RPM | 0 | Still flies up |
| SAC (PBRS) | RPM | 0 | Freezes |
| Curriculum | RPM | 0-1 | Stuck at stage 2 |
| **Velocity Control** | **VEL** | **2** | **Horizontal navigation!** |

### Why This Works

1. **Abstraction**: PID handles motor coordination, agent learns high-level navigation
2. **Intuitive actions**: [vx, vy, vz, yaw_rate] maps directly to intended movement
3. **Same as human pilots**: Swift/Nature paper uses thrust + body rates (similar abstraction)

### Lessons Learned

1. **Research first, code second** - spent too long trying to figure out drone RL from first principles
2. **Action space matters more than reward** - velocity control fixed what no reward tuning could
3. **Use established solutions** - gym-pybullet-drones already has velocity control built-in!

### Next Steps

1. Extend training to reach all 5 gates
2. Increase track complexity
3. Eventually return to direct motor control with imitation learning from velocity controller

### Research Sources

- [gym-pybullet-drones ActionType documentation](https://github.com/utiasDSL/gym-pybullet-drones)
- [Swift: Champion-level drone racing (Nature 2023)](https://www.nature.com/articles/s41586-023-06419-4)
- [Dream to Fly: Model-Based RL](https://arxiv.org/html/2501.14377v1)

---

## Performance Summary

| Version | Steps | Reward | Speed | Gates | Movement |
|---------|-------|--------|-------|-------|----------|
| Initial | 64 | -102 | 0.15 m/s | 0 | Crash |
| MAX_RPM fix | 875 | +307 | 5.30 m/s | 0 | Vertical |
| PPO v1 | 1000 | +930 | 8.8 m/s | 0 | Vertical |
| PPO v2 | 1000 | +84 | 9.6 m/s | 0 | Vertical |
| **Velocity Control** | **500** | **+18** | **~1 m/s** | **2** | **Horizontal** |

**First successful horizontal gate navigation achieved!**
