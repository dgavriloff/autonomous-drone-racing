# Competition Drone Specifications Research

## Summary

This document compares the Isaac Drone Racer simulation parameters with likely competition drone specs for the AI Grand Prix.

**Key Finding**: Competition drone specs (Neros Technologies) are not yet released. However, using A2RL and DCL racing specs as proxies, the current Isaac Drone Racer configuration is reasonably close but may need thrust-to-weight ratio adjustments.

---

## Competition Context

### AI Grand Prix (Anduril)
- **Drone Provider**: Neros Technologies (specs TBD)
- **Format**: Identical drones, software-only competition
- **Max Speed**: Expected ~150 km/h (based on A2RL proxy)
- **Sensors**: Single forward-facing camera + IMU (based on A2RL)
- **Control**: Direct motor commands, no human input

### A2RL x DCL Championship
- **Max Speed**: 150 km/h (41.7 m/s)
- **Track Size**: 27m x 35m indoor
- **Sensors**: Single camera + IMU for autonomous flight
- **Key Achievement**: AI beat world champion human pilot (2025)

---

## Current Isaac Drone Racer Configuration

From `tasks/drone_racer/mdp/actions.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Mass** | 0.6076 kg | Set in USD file |
| **Arm Length** | 0.035 m | 70mm motor-to-motor diagonal |
| **Thrust Coefficient** | 2.25e-7 | Per motor |
| **Drag Coefficient** | 1.5e-9 | Torque coefficient |
| **Omega Max** | 5145 rad/s | 49,140 RPM |
| **Motor KV** | 1950 | 6S LiPo (4.2V/cell) |
| **Thrust-to-Weight** | 4:1 | As designed |

### Calculated Performance
```
Max thrust per motor = thrust_coef × omega_max²
                     = 2.25e-7 × 5145²
                     = 5.96 N per motor

Total thrust (4 motors) = 23.84 N
Weight = 0.6076 × 9.81 = 5.96 N
Thrust-to-Weight = 23.84 / 5.96 = 4.0:1 ✓
```

---

## Competition Drone Estimates (A2RL/DCL Proxy)

Based on research from [GetFPV](https://www.getfpv.com/learn/new-to-fpv/fpv-drone-spec-racing-in-2024/), [Oscar Liang](https://oscarliang.com/table-prop-motor-lipo-weight/), and [DCL](https://dronechampionsleague.com/drones/):

### Typical 5-inch Racing Drone Specs
| Parameter | Range | Typical |
|-----------|-------|---------|
| **Mass** | 300-650g | 450-550g |
| **Prop Size** | 5-6 inch | 5 inch |
| **Motor Size** | 2207-2306 | 2207 |
| **Motor KV** | 1700-2200 | 1900 (6S) |
| **Battery** | 6S LiPo | 6S (25.2V) |
| **Thrust/Motor** | 1200-1600g | 1400g |
| **Thrust-to-Weight** | 4:1 to 8:1 | 6:1 |
| **Max Speed** | 140-160 km/h | 150 km/h |

### A2RL Specific Requirements
- Forward-facing camera only
- Onboard compute for autonomy
- Indoor racing (compact track)
- Speed up to 150 km/h (41.7 m/s)

---

## Gap Analysis

| Spec | Isaac Drone Racer | Competition Estimate | Gap |
|------|-------------------|---------------------|-----|
| Mass | 607g | 450-550g | +10-35% heavier |
| Thrust:Weight | 4:1 | 6:1 | 50% less power |
| Max Speed | ~25 m/s | ~42 m/s | 40% slower |
| Motor KV | 1950 | 1900-2100 | Close |
| Prop Size | 5" | 5-6" | Close |

---

## Recommended Configuration Changes

### Option 1: Increase Thrust (Keep Mass)
To achieve 6:1 thrust-to-weight with current mass:

```python
# Target: 6:1 thrust-to-weight
# Required total thrust = 6 × 0.6076 × 9.81 = 35.75 N
# Per motor = 8.94 N

# Method A: Increase thrust coefficient
# thrust_coef = 8.94 / omega_max² = 8.94 / 5145² = 3.38e-7
thrust_coef: float = 3.38e-7  # was 2.25e-7

# Method B: Increase omega_max (higher KV motor)
# omega_max = sqrt(8.94 / 2.25e-7) = 6303 rad/s
omega_max: float = 6303.0  # was 5145.0 (2400KV equivalent)
```

### Option 2: Reduce Mass (Keep Thrust)
To achieve 6:1 with current thrust:

```python
# Current max thrust = 23.84 N
# For 6:1, mass = 23.84 / (6 × 9.81) = 0.405 kg
# Would require modifying USD file mass to ~405g
```

### Option 3: Match A2RL Performance Target
For 150 km/h (41.7 m/s) capability:

```python
# Competition-like configuration
mass: float = 0.50  # 500g typical racing drone
thrust_coef: float = 3.38e-7  # Higher efficiency props
omega_max: float = 5500.0  # 2100KV motor, 6S
# Results in ~7:1 thrust-to-weight
```

---

## Implementation Steps

### 1. Modify Action Config
Edit `tasks/drone_racer/mdp/actions.py`:
```python
@configclass
class ControlActionCfg(ActionTermCfg):
    # ... existing params ...

    # MODIFIED for competition-like performance
    thrust_coef: float = 3.38e-7  # was 2.25e-7
    omega_max: float = 5500.0  # was 5145.0
```

### 2. Update USD File (if changing mass)
The mass is likely defined in:
- `assets/5_in_drone/5_in_drone.usd`

Would need to modify `mass` property in the rigid body.

### 3. Retrain with New Parameters
After config changes, retrain from scratch as dynamics will be different.

---

## Notes

1. **Neros Technologies specs not released** - These are estimates based on similar racing drones
2. **A2RL is best proxy** - Same DCL partnership, similar format
3. **Start conservative** - Current 4:1 ratio is reasonable for initial training
4. **Domain randomization** - Consider training with varied thrust-to-weight ratios for robustness

---

## Sources

- [AI Grand Prix Official](https://theaigrandprix.com/)
- [A2RL Autonomous Drone Racing](https://a2rl.io/autonomous-drone-race)
- [DCL Drones](https://dronechampionsleague.com/drones/)
- [GetFPV Spec Racing Guide](https://www.getfpv.com/learn/new-to-fpv/fpv-drone-spec-racing-in-2024/)
- [Oscar Liang Motor/Prop Guide](https://oscarliang.com/table-prop-motor-lipo-weight/)
- [The National - A2RL Coverage](https://www.thenationalnews.com/arts-culture/pop-culture/2026/01/20/umex-abu-dhabi-drone-racing-adnec/)
