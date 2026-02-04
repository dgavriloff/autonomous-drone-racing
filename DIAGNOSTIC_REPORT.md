# Vision Drone Racing - Diagnostic Report

**Date:** 2026-02-02
**Purpose:** Debug architecture mismatch between BC model and PPO fine-tuning

---

## 1. Architecture Dump

### 1.1 VisionStudentNetV2 (BC Model)

**File:** `scripts/train_vision_student.py` (lines 121-195)

```python
class VisionStudentNetV2(nn.Module):
    def __init__(
        self,
        gatenet_path: str = None,
        action_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (256, 256),
        freeze_encoder: bool = True,
        device: str = "cpu",
        num_frames: int = 1,
    ):
        super().__init__()

        # Encoder: 4 Conv2d layers with stride=2
        in_channels = 3 * num_frames  # 12 for 4-frame stacking
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 48x64 -> 24x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # 24x32 -> 12x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),          # 12x16 -> 6x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),         # 6x8 -> 3x4
            nn.ReLU(),
            nn.Flatten(),  # Output: 256 * 3 * 4 = 3072
        )

        # Policy head: 2 hidden layers WITH LayerNorm
        # For hidden_dims=(256, 256):
        self.policy_head = nn.Sequential(
            nn.Linear(3072, 256),   # policy_head.0
            nn.ReLU(),              # policy_head.1
            nn.LayerNorm(256),      # policy_head.2  <-- KEY DIFFERENCE
            nn.Linear(256, 256),    # policy_head.3
            nn.ReLU(),              # policy_head.4
            nn.LayerNorm(256),      # policy_head.5  <-- KEY DIFFERENCE
            nn.Linear(256, 4),      # policy_head.6
            nn.Tanh(),              # policy_head.7  <-- Bounded output [-1, 1]
        )
```

**State Dict Keys (from training log):**
```
encoder.0.weight: torch.Size([32, 12, 3, 3])
encoder.0.bias: torch.Size([32])
encoder.2.weight: torch.Size([64, 32, 3, 3])
encoder.2.bias: torch.Size([64])
encoder.4.weight: torch.Size([128, 64, 3, 3])
encoder.4.bias: torch.Size([128])
encoder.6.weight: torch.Size([256, 128, 3, 3])
encoder.6.bias: torch.Size([256])
policy_head.0.weight: torch.Size([256, 3072])
policy_head.0.bias: torch.Size([256])
policy_head.2.weight: torch.Size([256])        # LayerNorm gamma
policy_head.2.bias: torch.Size([256])          # LayerNorm beta
policy_head.3.weight: torch.Size([256, 256])
policy_head.3.bias: torch.Size([256])
policy_head.5.weight: torch.Size([256])        # LayerNorm gamma
policy_head.5.bias: torch.Size([256])          # LayerNorm beta
policy_head.6.weight: torch.Size([4, 256])
policy_head.6.bias: torch.Size([4])
```

**Input shape:** `(batch, 12, 48, 64)` - 4 stacked RGB frames, channels-first
**Output shape:** `(batch, 4)` - velocity commands in [-1, 1]
**Total params:** ~900K (estimated)

---

### 1.2 CustomCNN + PPO Policy (RL Model)

**File:** `scripts/rl_finetune_vision.py` (lines 143-174, 349-370)

```python
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 12

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # 3072
        )

        self.linear = nn.Sequential(
            nn.Linear(3072, 256),  # features_extractor.linear.0
            nn.ReLU(),             # features_extractor.linear.1
            # NO LayerNorm here!
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

# PPO policy_kwargs
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[256, 256], vf=[256, 256]),  # 2 hidden layers EACH
)
```

**PPO Policy State Dict Keys (from training log):**
```
log_std: torch.Size([4])

# Shared features extractor (may not be used)
features_extractor.cnn.0.weight: torch.Size([32, 12, 3, 3])
features_extractor.cnn.0.bias: torch.Size([32])
features_extractor.cnn.2.weight: torch.Size([64, 32, 3, 3])
features_extractor.cnn.2.bias: torch.Size([64])
features_extractor.cnn.4.weight: torch.Size([128, 64, 3, 3])
features_extractor.cnn.4.bias: torch.Size([128])
features_extractor.cnn.6.weight: torch.Size([256, 128, 3, 3])
features_extractor.cnn.6.bias: torch.Size([256])
features_extractor.linear.0.weight: torch.Size([256, 3072])
features_extractor.linear.0.bias: torch.Size([256])

# Policy-specific features extractor
pi_features_extractor.cnn.0.weight: torch.Size([32, 12, 3, 3])
pi_features_extractor.cnn.0.bias: torch.Size([32])
pi_features_extractor.cnn.2.weight: torch.Size([64, 32, 3, 3])
pi_features_extractor.cnn.2.bias: torch.Size([64])
pi_features_extractor.cnn.4.weight: torch.Size([128, 64, 3, 3])
pi_features_extractor.cnn.4.bias: torch.Size([128])
pi_features_extractor.cnn.6.weight: torch.Size([256, 128, 3, 3])
pi_features_extractor.cnn.6.bias: torch.Size([256])
pi_features_extractor.linear.0.weight: torch.Size([256, 3072])
pi_features_extractor.linear.0.bias: torch.Size([256])

# Value-specific features extractor
vf_features_extractor.cnn.0.weight: torch.Size([32, 12, 3, 3])
vf_features_extractor.cnn.0.bias: torch.Size([32])
vf_features_extractor.cnn.2.weight: torch.Size([64, 32, 3, 3])
vf_features_extractor.cnn.2.bias: torch.Size([64])
vf_features_extractor.cnn.4.weight: torch.Size([128, 64, 3, 3])
vf_features_extractor.cnn.4.bias: torch.Size([128])
vf_features_extractor.cnn.6.weight: torch.Size([256, 128, 3, 3])
vf_features_extractor.cnn.6.bias: torch.Size([256])
vf_features_extractor.linear.0.weight: torch.Size([256, 3072])
vf_features_extractor.linear.0.bias: torch.Size([256])

# MLP layers AFTER features (net_arch=dict(pi=[256, 256]))
mlp_extractor.policy_net.0.weight: torch.Size([256, 256])
mlp_extractor.policy_net.0.bias: torch.Size([256])
mlp_extractor.policy_net.2.weight: torch.Size([256, 256])  # <-- NO BC EQUIVALENT
mlp_extractor.policy_net.2.bias: torch.Size([256])         # <-- NO BC EQUIVALENT
mlp_extractor.value_net.0.weight: torch.Size([256, 256])
mlp_extractor.value_net.0.bias: torch.Size([256])
mlp_extractor.value_net.2.weight: torch.Size([256, 256])
mlp_extractor.value_net.2.bias: torch.Size([256])

# Final output layers
action_net.weight: torch.Size([4, 256])
action_net.bias: torch.Size([4])
value_net.weight: torch.Size([1, 256])
value_net.bias: torch.Size([1])
```

**Input shape:** `(batch, 12, 48, 64)` - same as BC
**Output shape:** `(batch, 4)` - Gaussian mean (NOT bounded by Tanh)

---

## 2. Pipeline Overview

### 2.1 Current Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TEACHER (State-Based)                            │
│  models/curriculum_final.zip (PPO, 5/5 gates)                          │
│  Input: 12D state [pos, vel, quat, angular_vel, gate_rel]              │
│  Output: 4D velocity [vx, vy, vz, yaw_rate]                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DEMO COLLECTION (collect_teacher_demos.py)           │
│  - Run teacher in CameraRacingEnv                                       │
│  - Save: RGB images (64x48) + teacher actions                           │
│  - Optional: DART noise injection (record clean action)                 │
│  - Output: data/teacher_demos/ or data/dagger/iter_XX/                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    BC TRAINING (train_vision_student.py)                │
│  - Input: (image, action) pairs                                         │
│  - Model: VisionStudentNetV2                                            │
│  - Loss: MSE(predicted_action, teacher_action)                          │
│  - Output: data/dagger/iter_XX/model/best_model.pt                      │
│  - Performance: 3.2/5 gates (standalone evaluation)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RL FINE-TUNING (rl_finetune_vision.py)               │
│  - Load BC weights into PPO (ARCHITECTURE MISMATCH HERE)                │
│  - Train with PPO on VisionRacingEnv                                    │
│  - Reward: +100 per gate, progress bonus, -0.1 time penalty             │
│  - Output: models/vision_rl/best_model.zip                              │
│  - CURRENT ISSUE: Only 1/5 gates after weight transfer                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Detailed Forward Pass Comparison

**BC Model Forward Pass:**
```
Input: (B, 12, 48, 64)
  │
  ▼
encoder.0 (Conv2d 12→32, stride=2) → ReLU
  │ Output: (B, 32, 24, 32)
  ▼
encoder.2 (Conv2d 32→64, stride=2) → ReLU
  │ Output: (B, 64, 12, 16)
  ▼
encoder.4 (Conv2d 64→128, stride=2) → ReLU
  │ Output: (B, 128, 6, 8)
  ▼
encoder.6 (Conv2d 128→256, stride=2) → ReLU
  │ Output: (B, 256, 3, 4)
  ▼
Flatten
  │ Output: (B, 3072)
  ▼
policy_head.0 (Linear 3072→256) → ReLU → LayerNorm ← NOT IN PPO
  │ Output: (B, 256) normalized
  ▼
policy_head.3 (Linear 256→256) → ReLU → LayerNorm  ← NOT IN PPO
  │ Output: (B, 256) normalized
  ▼
policy_head.6 (Linear 256→4) → Tanh                ← PPO uses Gaussian
  │ Output: (B, 4) bounded [-1, 1]
```

**PPO Model Forward Pass:**
```
Input: (B, 12, 48, 64)
  │
  ▼
pi_features_extractor.cnn.0 (Conv2d 12→32) → ReLU
  │
  ▼
pi_features_extractor.cnn.2 (Conv2d 32→64) → ReLU
  │
  ▼
pi_features_extractor.cnn.4 (Conv2d 64→128) → ReLU
  │
  ▼
pi_features_extractor.cnn.6 (Conv2d 128→256) → ReLU
  │
  ▼
Flatten → (B, 3072)
  │
  ▼
pi_features_extractor.linear.0 (Linear 3072→256) → ReLU  ← NO LayerNorm
  │ Output: (B, 256) NOT normalized
  ▼
mlp_extractor.policy_net.0 (Linear 256→256) → ReLU       ← NO LayerNorm
  │ Output: (B, 256) NOT normalized
  ▼
mlp_extractor.policy_net.2 (Linear 256→256) → ReLU       ← NO BC EQUIVALENT
  │ Output: (B, 256) EXTRA LAYER
  ▼
action_net (Linear 256→4)                                 ← NO Tanh
  │ Output: (B, 4) Gaussian mean, unbounded
  ▼
Sample from N(mean, exp(log_std))
  │ Output: (B, 4) sampled action
```

---

## 3. Training Setup

### 3.1 BC Training (Behavioral Cloning)

| Parameter | Value |
|-----------|-------|
| Algorithm | Supervised Learning (MSE loss) |
| Observation | RGB image (12, 48, 64) - 4 stacked frames |
| Action | 4D velocity [-1, 1] |
| Dataset | ~40K frames from teacher demos |
| Epochs | 20 |
| Batch size | 256 |
| Learning rate | 3e-4 with cosine annealing |
| Augmentation | Brightness, noise |
| Best val loss | ~0.02 |
| **Performance** | **3.2/5 gates** (standalone eval) |

### 3.2 RL Fine-tuning (PPO)

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Observation | RGB image (12, 48, 64) - 4 stacked frames |
| Action | 4D velocity, Gaussian distribution |
| Parallel envs | 16 (SubprocVecEnv) |
| n_steps | 2048 |
| batch_size | 256 |
| n_epochs | 10 |
| learning_rate | 3e-4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| Total timesteps | 500,000 |
| **Performance** | **1/5 gates** (after weight transfer) |

### 3.3 Reward Function (VisionRacingEnv)

```python
def _compute_reward(self, info, obs):
    reward = 0.0

    # Gate passed: +100
    if current_gate > self.prev_gate:
        reward += 100.0

    # Progress toward gate: distance_reduction * 10
    progress = self.prev_dist - dist
    reward += progress * 10.0

    # Time penalty: -0.1 per step
    reward -= 0.1

    # Crash penalty: -50
    if info.get("crashed", False):
        reward -= 50.0

    return reward
```

---

## 4. Weight Transfer Analysis

### 4.1 Current Transfer Mapping

| BC Layer | Shape | PPO Layer | Shape | Status |
|----------|-------|-----------|-------|--------|
| encoder.0.weight | (32, 12, 3, 3) | pi_features_extractor.cnn.0.weight | (32, 12, 3, 3) | ✓ Match |
| encoder.0.bias | (32,) | pi_features_extractor.cnn.0.bias | (32,) | ✓ Match |
| encoder.2.weight | (64, 32, 3, 3) | pi_features_extractor.cnn.2.weight | (64, 32, 3, 3) | ✓ Match |
| encoder.2.bias | (64,) | pi_features_extractor.cnn.2.bias | (64,) | ✓ Match |
| encoder.4.weight | (128, 64, 3, 3) | pi_features_extractor.cnn.4.weight | (128, 64, 3, 3) | ✓ Match |
| encoder.4.bias | (128,) | pi_features_extractor.cnn.4.bias | (128,) | ✓ Match |
| encoder.6.weight | (256, 128, 3, 3) | pi_features_extractor.cnn.6.weight | (256, 128, 3, 3) | ✓ Match |
| encoder.6.bias | (256,) | pi_features_extractor.cnn.6.bias | (256,) | ✓ Match |
| policy_head.0.weight | (256, 3072) | pi_features_extractor.linear.0.weight | (256, 3072) | ✓ Match |
| policy_head.0.bias | (256,) | pi_features_extractor.linear.0.bias | (256,) | ✓ Match |
| **policy_head.2.weight** | **(256,)** | **N/A** | **N/A** | **✗ LayerNorm missing in PPO** |
| **policy_head.2.bias** | **(256,)** | **N/A** | **N/A** | **✗ LayerNorm missing in PPO** |
| policy_head.3.weight | (256, 256) | mlp_extractor.policy_net.0.weight | (256, 256) | ✓ Match |
| policy_head.3.bias | (256,) | mlp_extractor.policy_net.0.bias | (256,) | ✓ Match |
| **policy_head.5.weight** | **(256,)** | **N/A** | **N/A** | **✗ LayerNorm missing in PPO** |
| **policy_head.5.bias** | **(256,)** | **N/A** | **N/A** | **✗ LayerNorm missing in PPO** |
| policy_head.6.weight | (4, 256) | action_net.weight | (4, 256) | ✓ Match |
| policy_head.6.bias | (4,) | action_net.bias | (4,) | ✓ Match |
| N/A | N/A | **mlp_extractor.policy_net.2.weight** | **(256, 256)** | **✗ Extra layer in PPO** |
| N/A | N/A | **mlp_extractor.policy_net.2.bias** | **(256,)** | **✗ Extra layer in PPO** |

### 4.2 Critical Mismatches

1. **LayerNorm layers (policy_head.2, policy_head.5):**
   - BC normalizes activations after each ReLU
   - PPO does not have LayerNorm
   - Result: Weights trained with normalized inputs receive unnormalized inputs

2. **Extra hidden layer (mlp_extractor.policy_net.2):**
   - BC has 2 hidden layers: 3072→256→256→4
   - PPO has 3 hidden layers: 3072→256→256→256→4
   - Result: action_net receives input from wrong layer

3. **Output activation:**
   - BC uses Tanh (bounded [-1, 1])
   - PPO outputs Gaussian mean (unbounded), then samples
   - Result: Different action distributions

---

## 5. Current Problem

### 5.1 Observed Behavior

- **BC model standalone:** 3.2/5 gates average
- **After weight transfer to PPO:** 1/5 gates average
- **Expected:** Should maintain ~3/5 gates initially, then improve with RL

### 5.2 Training Log Output

```
Loading BC weights from: data/dagger/iter_02/model/best_model.pt
  ✓ encoder.0.weight -> pi_features_extractor.cnn.0.weight
  ✓ encoder.0.bias -> pi_features_extractor.cnn.0.bias
  ✓ encoder.2.weight -> pi_features_extractor.cnn.2.weight
  ... (8 encoder weights)
  ✓ policy_head.0.weight -> features_extractor.linear.0.weight
  ✓ policy_head.0.bias -> features_extractor.linear.0.bias
  ✓ policy_head.3.weight -> mlp_extractor.policy_net.0.weight
  ✓ policy_head.3.bias -> mlp_extractor.policy_net.0.bias
  ✓ policy_head.6.weight -> action_net.weight
  ✓ policy_head.6.bias -> action_net.bias
  ... (encoder to pi_features_extractor)

Loaded 24 weight tensors from BC checkpoint

--- Initial Evaluation (before training) ---
Gates passed: [1, 1, 1, 1, 1], avg: 1.0/5
```

### 5.3 Root Cause

The weight transfer appears successful (shapes match), but the **computational graph is different**:

1. BC layer `policy_head.3` expects **LayerNorm-normalized** input from layer 0
2. PPO layer `mlp_extractor.policy_net.0` receives **unnormalized** input
3. The learned weights are meaningless in this context

---

## 6. File Locations

| Description | Path |
|-------------|------|
| BC model definition | `scripts/train_vision_student.py` (VisionStudentNetV2, line 121) |
| PPO model definition | `scripts/rl_finetune_vision.py` (CustomCNN, line 143) |
| BC training script | `scripts/train_vision_student.py` |
| RL training script | `scripts/rl_finetune_vision.py` |
| Demo collection | `scripts/collect_teacher_demos.py` |
| Teacher model | `models/curriculum_final.zip` |
| BC checkpoint | `data/dagger/iter_02/model/best_model.pt` (on remote PC) |
| RL output | `models/vision_rl/best_model.zip` |
| Training log | `/tmp/rl_finetune.log` (on remote PC) |

---

## 7. Potential Solutions

### Option A: Retrain BC without LayerNorm
- Modify `train_vision_student.py` to remove LayerNorm from policy_head
- Retrain BC model
- Weight transfer should work cleanly
- **Risk:** BC training may be less stable

### Option B: Add LayerNorm to PPO
- Create custom MLP extractor class with LayerNorm
- Match BC architecture exactly
- More complex SB3 customization required

### Option C: Use BC model directly (no PPO)
- Wrap BC model in a policy class
- Use DAgger or direct fine-tuning instead of PPO
- Avoids architecture conversion issues

### Option D: Reduce PPO layers
- Change `net_arch=dict(pi=[256], vf=[256, 256])` - only 1 hidden layer in policy
- Still has LayerNorm mismatch but fewer layers to debug

---

## 8. Questions for Debugger

1. Is the LayerNorm truly critical, or can the network adapt?
2. Should we prioritize architecture matching or simpler training?
3. Is there a way to convert LayerNorm weights to equivalent bias adjustments?
4. Would a different RL algorithm (SAC, TD3) have the same issues?
