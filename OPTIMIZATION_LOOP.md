# Optimization Loop for Isaac Drone Racer

**Purpose:** Systematic iteration protocol that survives context compactions.

---

## Quick Start (For Claude After Compaction)

```bash
# 1. Check current state
./scripts/remote/training-status.sh

# 2. Read the experiment queue
cat OPTIMIZATION_LOOP.md  # (this file)

# 3. Continue from CURRENT EXPERIMENT section below
```

---

## The Loop

```
┌─────────────────────────────────────────────────────────┐
│  1. CHECK STATE                                         │
│     - What's running? What's the best result so far?    │
├─────────────────────────────────────────────────────────┤
│  2. RUN EXPERIMENT                                      │
│     - Start training with current config                │
│     - Wait for completion (~1hr for 50K iters)          │
├─────────────────────────────────────────────────────────┤
│  3. EVALUATE                                            │
│     - Extract TensorBoard metrics                       │
│     - Compare gate_passed to baseline                   │
├─────────────────────────────────────────────────────────┤
│  4. DECIDE                                              │
│     - gate_passed > baseline+0.3 → KEEP, update baseline│
│     - gate_passed ≈ baseline → COMBINE with next idea   │
│     - gate_passed < baseline-0.3 → REVERT               │
├─────────────────────────────────────────────────────────┤
│  5. LOG & ITERATE                                       │
│     - Record result in EXPERIMENT LOG below             │
│     - Pick next experiment from QUEUE                   │
│     - Go to step 2                                      │
└─────────────────────────────────────────────────────────┘
```

---

## Current Baseline

| Metric | Value | Run |
|--------|-------|-----|
| **gate_passed (best)** | 4.21 | 2026-02-01_13-18-41 |
| **gate_passed (final)** | 4.08 | 2026-02-01_13-18-41 |
| **config** | 6:1 thrust, 50K iters, 4096 envs | |
| **speed** | ~19.6 m/s (71 km/h) | |

**Target:** 7/7 gates (or 5/5 on simplified track)

---

## CURRENT EXPERIMENT

**Status:** DISCOVERY - curriculum already enabled!

**Finding:** `randomise_start=True` is already set, which:
- Randomly picks starting gate (0-6)
- Spawns drone near previous gate
- Agent already sees all gates during training

**Why we're still stuck at 4.15 gates:**
1. Random curriculum may not focus enough on hard gates
2. Later gates have different geometry (heights, yaw angles)
3. Gate reward (400) may not be strong enough vs crash penalty (-500)

**Next experiment:** #3 - Increase gate reward to 800

**Rationale:** Make passing each gate more valuable. Current ratio is 400:500 (gate:crash). Try 800:500 to make gates 1.6x more valuable than avoiding crashes.

**How to do it:**
```bash
# Edit the config file on remote
ssh training-pc 'wsl bash -c "sed -i \"s/gate_passed, weight=400/gate_passed, weight=800/\" ~/repos/isaac_drone_racer/tasks/drone_racer/drone_racer_env_cfg.py"'

# Verify change
ssh training-pc 'wsl grep gate_passed ~/repos/isaac_drone_racer/tasks/drone_racer/drone_racer_env_cfg.py'

# Start training
./scripts/remote/start-training.sh 50000 4096
```

---

## Experiment Queue

| # | Experiment | Hypothesis | Config Change |
|---|------------|------------|---------------|
| 1 | **100K iterations** | More training = more gates | `--max_iterations 100000` |
| 2 | **Curriculum: start at gate 3** | Agent never sees late gates | Modify env to start mid-track |
| 3 | **Higher thrust (8:1)** | More power = more agility | `thrust_coef=2.55e-7, omega_max=5700` |
| 4 | **Reward: increase gate bonus** | Stronger signal for gates | `gate_passed=800` (was 400) |
| 5 | **Simpler track (5 gates)** | Easier problem first | Modify track config |
| 6 | **Reduce num_envs to 2048** | More gradient steps per sample | `--num_envs 2048` |

---

## Experiment Log

### Experiment #0: Baseline (6:1 thrust, 50K)
- **Date:** 2026-02-01
- **Run:** 2026-02-01_13-18-41_ppo_torch
- **Config:** 6:1 thrust, 50K iters, 4096 envs, 7-gate track
- **Result:** gate_passed=4.21 (best), 4.08 (final)
- **Decision:** BASELINE ESTABLISHED

### Experiment #1: 100K iterations
- **Date:** 2026-02-01
- **Run:** 2026-02-01_15-15-38_ppo_torch
- **Config:** Same as baseline, 100K iters (ran to 240K steps)
- **Result:** gate_passed=4.15 (final), 4.15 (max) - PLATEAU
- **Comparison:** Baseline max was 4.21, this never exceeded 4.15
- **Decision:** ⚠️ COMBINE - more training alone doesn't help, need curriculum or reward changes

### Experiment #2: Higher gate reward (800 vs 400)
- **Date:** 2026-02-02
- **Run:** STARTED 19:26 PST
- **Config:** gate_passed reward=800 (was 400), 50K iters, 4096 envs
- **Hypothesis:** Higher gate reward (1.6x crash penalty) will incentivize gate completion
- **Result:** PENDING - training ~45-60 minutes
- **Discovery:** `randomise_start=True` already provides curriculum - drone spawns at random gates

---

## Metric Extraction Commands

### Quick metrics check
```bash
ssh training-pc 'wsl bash -c "cd ~/repos/isaac_drone_racer && python -c \"
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator
import sys
run = sys.argv[1] if len(sys.argv) > 1 else '2026-02-01_15-15-38_ppo_torch'
ea = event_accumulator.EventAccumulator(f'logs/skrl/drone_racer/{run}')
ea.Reload()
for tag in ['Info / Episode_Reward/gate_passed', 'Reward / Total reward (mean)']:
    try:
        scalars = ea.Scalars(tag)
        print(f'{tag}: {scalars[-1].value:.2f} (step {scalars[-1].step})')
    except: pass
\""'
```

### Full metrics dump
```bash
ssh training-pc 'wsl bash -c "cd ~/repos/isaac_drone_racer && python -c \"
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator('logs/skrl/drone_racer/RUNNAME')
ea.Reload()
print('Available tags:', ea.Tags()['scalars'][:10])
\""'
```

---

## Decision Criteria

| gate_passed | vs Baseline | Action |
|-------------|-------------|--------|
| > 4.5 | +0.3 better | ✅ KEEP as new baseline |
| 4.0 - 4.5 | ~same | ⚠️ COMBINE with next experiment |
| < 4.0 | worse | ❌ REVERT to baseline config |
| > 6.0 | breakthrough | 🎉 MAJOR WIN - document everything |

---

## Files to Know

| File | Purpose |
|------|---------|
| `OPTIMIZATION_LOOP.md` | This file - iteration protocol |
| `USING_TRAINING_PC.md` | SSH commands and remote scripts |
| `blogs.md` | Development history and discoveries |
| `CLAUDE.md` | Project overview |
| `scripts/remote/*.sh` | Helper scripts for training PC |

---

## Recovery After Compaction

If context was compacted, Claude should:

1. **Read this file** to understand current state
2. **Check training status:** `./scripts/remote/training-status.sh`
3. **Look at CURRENT EXPERIMENT section** for what to do next
4. **Continue the loop**

---

## Notes

- Each 50K iteration run takes ~45-60 minutes on RTX 5080
- 4096 envs uses ~6GB VRAM, leaves headroom
- Always use tmux for training (survives SSH disconnect)
- Checkpoints save every 5K iterations
