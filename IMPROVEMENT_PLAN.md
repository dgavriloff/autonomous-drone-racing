# Isaac Drone Racer Improvement Plan

## Current Baseline (6:1 thrust-to-weight)
| Metric | Value |
|--------|-------|
| Gates Passed | 4.08 avg, 4.21 best |
| Total Reward | 115.17 avg |
| Episode Length | 1620 steps |
| Training Time | ~59 min / 50K iterations |

**Goal**: Reach 5+ gates consistently, then full track completion

---

## Experiment Queue

### Experiment 1: Longer Training
**Hypothesis**: Agent may still be improving at 50K, more iterations could help

```bash
# Run 100K iterations
python scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --num_envs 4096 --max_iterations 100000
```

**Success metric**: gate_passed > 4.5

---

### Experiment 2: Higher Thrust-to-Weight (8:1)
**Hypothesis**: Competition drones run 6:1 to 8:1, more power = more agility

**Config changes** (`tasks/drone_racer/mdp/actions.py`):
```python
# 8:1 thrust-to-weight
thrust_coef: float = 3.20e-7  # was 2.40e-7
omega_max: float = 5540.0     # keep same
init: list[float] = (1958.0, 1958.0, 1958.0, 1958.0)  # recalculated hover
```

**Math**:
- Target thrust = 8 × 0.6076 × 9.81 = 47.67 N total
- Per motor = 11.92 N
- thrust_coef = 11.92 / omega_max² = 11.92 / 5540² = 3.88e-7 (or increase omega_max)

**Success metric**: gate_passed > 4.5

---

### Experiment 3: Reward Tuning
**Hypothesis**: Current rewards may not incentivize completing later gates

**Current rewards** (from env config):
- gate_passed: 400
- progress: 20
- terminating: -500

**Try A - Increase gate reward**:
- gate_passed: 600 (50% increase)
- progress: 20
- terminating: -500

**Try B - Reduce termination penalty**:
- gate_passed: 400
- progress: 20
- terminating: -300 (less fear of crashing = more aggressive)

**Success metric**: gate_passed > 4.5

---

### Experiment 4: Curriculum Learning on Gate Count
**Hypothesis**: Agent never sees gate 5+ because it crashes before reaching them

**Approach**: Start with 3 gates, add more as agent improves
1. Train on 3 gates until gate_passed > 2.8
2. Increase to 4 gates until gate_passed > 3.8
3. Increase to 5 gates until gate_passed > 4.8
4. Continue to full track

**Implementation**: Modify env to accept `num_gates` parameter

**Success metric**: Reach 5/5 gates

---

### Experiment 5: Domain Randomization
**Hypothesis**: Agent overfits to specific track conditions

**Randomize**:
- Gate positions (±0.5m)
- Drone mass (±10%)
- Thrust coefficient (±10%)
- Wind/disturbances

**Success metric**: More robust performance, gate_passed maintains > 4.0

---

## Execution Order

| Priority | Experiment | Rationale |
|----------|------------|-----------|
| 1 | Longer Training (100K) | Cheapest test, just more compute |
| 2 | 8:1 Thrust | Simple config change, validated approach |
| 3 | Curriculum Learning | Addresses root cause of not seeing later gates |
| 4 | Reward Tuning | May help but harder to debug |
| 5 | Domain Randomization | For robustness after basic perf achieved |

---

## Metrics to Track

For each experiment, extract from TensorBoard:
```python
metrics = [
    'Info / Episode_Reward/gate_passed',      # Primary metric
    'Reward / Total reward (mean)',            # Overall performance
    'Episode / Total timesteps (mean)',        # Survival time
    'Info / Episode_Reward/terminating',       # Crash rate
    'Info / Episode_Reward/progress',          # Forward motion
    'Loss / Policy loss',                      # Training stability
]
```

---

## Quick Iteration Script

```bash
# Run experiment and extract metrics
ssh ooousay@100.99.201.43 'wsl -e bash -c "
  cd ~/repos/isaac_drone_racer
  source ~/repos/isaac-racing-venv/bin/activate

  # Train
  python scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --num_envs 4096

  # Get latest run
  LATEST=\$(ls -t logs/skrl/drone_racer/ | head -1)
  echo \"Run: \$LATEST\"
"'

# Then extract metrics with the TensorBoard script
```

---

## Decision Criteria

After each experiment:
1. If gate_passed > 4.5: **Keep change, continue**
2. If gate_passed ≈ 4.0-4.5: **Combine with next experiment**
3. If gate_passed < 4.0: **Revert, try different approach**

---

## Notes

- WSL2 can't run visual evaluation (Vulkan issue)
- Use TensorBoard metrics extraction instead
- Each 50K run takes ~1 hour on RTX 5080
- Can run 2 experiments overnight
