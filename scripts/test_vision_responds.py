#!/usr/bin/env python3
"""Quick test to verify the model responds to different visual inputs."""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train_vision_student import VisionStudentNetV2

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VisionStudentNetV2(num_frames=4, device=device).to(device)
ckpt = torch.load('models/vision_student/best_model.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print("Testing if model responds to different visual inputs...\n")

# Test inputs
inputs = {
    'black': torch.zeros(1, 12, 48, 64),
    'white': torch.ones(1, 12, 48, 64),
    'random1': torch.rand(1, 12, 48, 64),
    'random2': torch.rand(1, 12, 48, 64),
}

actions = {}
with torch.no_grad():
    for name, img in inputs.items():
        action = model(img.to(device)).cpu().numpy().squeeze()
        actions[name] = action
        print(f"{name:10}: [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}, {action[3]:+.3f}]")

print("\n--- Analysis ---")
print(f"Black vs White different: {not np.allclose(actions['black'], actions['white'], atol=0.05)}")
print(f"Random1 vs Random2 different: {not np.allclose(actions['random1'], actions['random2'], atol=0.05)}")
print(f"Black vs Random different: {not np.allclose(actions['black'], actions['random1'], atol=0.05)}")

if not np.allclose(actions['black'], actions['white'], atol=0.05):
    print("\n✓ Model IS responding to visual input (not hardcoded)")
else:
    print("\n✗ Model might be ignoring visual input")
