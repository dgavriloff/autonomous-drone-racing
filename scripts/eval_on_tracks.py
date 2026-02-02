#!/usr/bin/env python3
"""
Evaluate vision student on multiple track configurations.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_vision_student import VisionStudentNetV2
from scripts.collect_teacher_demos import CameraRacingEnv
from scripts.train_parallel import create_simple_track, create_competition_track


def evaluate_on_track(model, track, num_episodes=5, max_steps=1000, device="cuda"):
    """Evaluate model on a specific track."""
    env = CameraRacingEnv(
        track=track,
        image_size=(64, 48),
        ctrl_freq=48,
        pyb_freq=240,
        gui=False,
        gate_tolerance=1.0,
        max_steps=max_steps,
    )

    num_frames = model.num_frames
    results = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        frame_buffer = []

        while not done:
            rgb = env.get_camera_image()
            rgb_norm = rgb.astype(np.float32) / 255.0
            frame_buffer.append(rgb_norm)
            if len(frame_buffer) > num_frames:
                frame_buffer.pop(0)

            if len(frame_buffer) < num_frames:
                padded = [frame_buffer[0]] * (num_frames - len(frame_buffer)) + frame_buffer
            else:
                padded = frame_buffer

            stacked = np.concatenate(padded, axis=-1)
            img = torch.from_numpy(stacked).float().permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(img).squeeze().cpu().numpy().reshape(1, -1)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        results.append(info.get("gates_passed", 0))

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate vision student on multiple tracks")
    parser.add_argument("--model", default="models/vision_student/best_model.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    num_frames = checkpoint.get("num_frames", 1)

    model = VisionStudentNetV2(
        action_dim=4,
        hidden_dims=(256, 256),
        device=device,
        num_frames=num_frames,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model (epoch {checkpoint.get('epoch', '?')}, num_frames={num_frames})")
    print(f"Device: {device}")
    print()

    # Define tracks to test
    tracks = [
        ("5-gate circle (baseline)", create_simple_track(num_gates=5, radius=1.5)),
        ("7-gate circle", create_simple_track(num_gates=7, radius=2.0)),
        ("Swift 7-gate", create_competition_track("swift", scale=0.3)),
        ("Figure-8", create_competition_track("figure8", scale=0.25)),
        ("TII 11-gate", create_competition_track("tii", scale=0.25)),
    ]

    print("=" * 60)
    print("MULTI-TRACK EVALUATION")
    print("=" * 60)
    print()

    for name, track in tracks:
        results = evaluate_on_track(model, track, num_episodes=args.episodes, device=device)
        num_gates = len(track.gates)
        avg = np.mean(results)
        best = max(results)
        print(f"{name}:")
        print(f"  Gates: {num_gates}, Avg passed: {avg:.1f}/{num_gates} ({100*avg/num_gates:.0f}%)")
        print(f"  Results: {results}, Best: {best}/{num_gates}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
