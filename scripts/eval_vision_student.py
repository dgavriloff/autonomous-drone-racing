#!/usr/bin/env python3
"""
Evaluate a trained vision student policy.

Runs the vision student in the racing environment using camera input only.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_vision_student import VisionStudentNetV2
from scripts.collect_teacher_demos import CameraRacingEnv
from scripts.train_parallel import create_simple_track


def evaluate_vision_student(
    model_path: str,
    num_episodes: int = 10,
    num_gates: int = 5,
    max_steps: int = 1000,
    render: bool = False,
    device: str = None,
):
    """
    Evaluate vision student policy.

    Args:
        model_path: Path to trained vision student model
        num_episodes: Number of evaluation episodes
        num_gates: Number of gates in track
        max_steps: Max steps per episode
        render: Whether to render (GUI)
        device: Device for inference
    """
    print("=" * 70)
    print("EVALUATING VISION STUDENT")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print()

    # Setup device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}")

    # Load model
    model = VisionStudentNetV2(
        action_dim=4,
        hidden_dims=(256, 256),
        device=device,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
    print()

    # Create environment
    track = create_simple_track(num_gates=num_gates, radius=1.5)
    env = CameraRacingEnv(
        track=track,
        image_size=(64, 48),
        ctrl_freq=48,
        pyb_freq=240,
        gui=render,
        gate_tolerance=1.0,
        max_steps=max_steps,
    )

    # Evaluate
    results = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Get camera image
            rgb = env.get_camera_image()

            # Preprocess: (H,W,C) -> (1,C,H,W), normalize
            img = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
            img = img / 255.0
            img = img.to(device)

            # Get action from vision student
            with torch.no_grad():
                action = model(img).squeeze().cpu().numpy()

            # Reshape for environment (expects 2D array)
            action = action.reshape(1, -1)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        gates_passed = info.get("gates_passed", 0)
        results.append({
            "episode": ep,
            "gates_passed": gates_passed,
            "total_reward": total_reward,
            "steps": steps,
            "success": gates_passed == num_gates,
        })

        print(f"Episode {ep+1}/{num_episodes}: {gates_passed}/{num_gates} gates, "
              f"reward={total_reward:.1f}, steps={steps}")

    env.close()

    # Summary
    avg_gates = np.mean([r["gates_passed"] for r in results])
    success_rate = np.mean([r["success"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Average gates passed: {avg_gates:.2f}/{num_gates}")
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Average reward: {avg_reward:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate vision student policy")
    parser.add_argument("--model", default="models/vision_student/best_model.pt",
                        help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--gates", type=int, default=5,
                        help="Number of gates")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per episode")
    parser.add_argument("--render", action="store_true",
                        help="Render with GUI")
    parser.add_argument("--device", default=None,
                        help="Device (cuda, mps, cpu)")
    args = parser.parse_args()

    evaluate_vision_student(
        model_path=args.model,
        num_episodes=args.episodes,
        num_gates=args.gates,
        max_steps=args.max_steps,
        render=args.render,
        device=args.device,
    )


if __name__ == "__main__":
    main()
