# Vision-Based Drone Training Plan

## Goal
Train a vision-only drone policy that can race through gates using only camera input.

## What We Have
- **Teacher Policy**: `models/curriculum_final.zip` - SAC policy, 5/5 gates, velocity control
- **GateNet**: `models/gate_net/best_model.pt` - U-Net segmentation, 76% IoU on TII Racing
- **Vision Pipeline**: GateNet → QuAdGate → PoseEstimator (tested end-to-end)

## Architecture

```
Camera (64x48 RGB)
    ↓
GateNet (U-Net, 482K params)
    ↓
Gate Segmentation Mask
    ↓
QuAdGate (corner detection)
    ↓
4 Corner Points + Confidence
    ↓
[Optional: PoseEstimator → 6-DoF gate pose]
    ↓
Vision Features (flattened)
    ↓
Student Policy (MLP)
    ↓
Velocity Commands [vx, vy, vz, yaw_rate]
```

## Training Pipeline

### Phase 1: Data Collection
1. Load teacher policy (privileged state → actions)
2. Run teacher in gym-pybullet-drones with camera enabled
3. For each timestep, save:
   - Camera RGB image (64x48)
   - Teacher action (4D velocity command)
   - Gate positions (for debugging)
   - Drone state (for debugging)
4. Target: 10,000+ successful gate passages

### Phase 2: Behavioral Cloning
1. Create dataset: (image, action) pairs
2. Student architecture:
   - Input: 64x48 RGB image
   - GateNet encoder (frozen or fine-tuned)
   - Policy head: MLP [256, 256] → 4D action
3. Training:
   - Loss: MSE(student_action, teacher_action)
   - Optimizer: Adam, lr=3e-4
   - Batch size: 256
   - Epochs: 50-100

### Phase 3: DAgger (Dataset Aggregation)
1. Run student policy, collect trajectories
2. At each state, also query teacher for action
3. When student fails (misses gate, crashes):
   - Add (image, teacher_action) to dataset
4. Retrain student on aggregated dataset
5. Repeat 3-5 iterations

### Phase 4: RL Fine-tuning (Optional)
1. Initialize SAC with BC-trained student
2. Fine-tune with low learning rate (1e-5)
3. Use same reward as teacher training
4. Early stop if performance degrades

## Implementation Files

### scripts/collect_vision_data.py
- Runs teacher policy with camera
- Saves demonstrations to disk

### scripts/train_vision_student.py
- Behavioral cloning training
- Uses GateNet as encoder

### scripts/dagger_training.py
- DAgger loop implementation
- Iterative data aggregation

### src/models/vision_student.py
- Student policy architecture
- GateNet integration

## Success Criteria
- [ ] Student achieves 3/5 gates with vision only
- [ ] Student achieves 5/5 gates with vision only
- [ ] Student matches teacher speed (within 20%)
- [ ] Works with domain randomization (lighting, texture)

## Notes
- Camera runs at 24 Hz, control at 48 Hz
- Use ground truth for initial testing, then pure vision
- GateNet can be frozen (transfer learning) or fine-tuned
