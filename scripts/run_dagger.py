#!/usr/bin/env python3
"""
DAgger (Dataset Aggregation) for vision-based drone racing.

Iteratively improves the student policy by:
1. Rolling out current student policy
2. Querying teacher for correct actions at student's visited states
3. Aggregating new data with existing dataset
4. Retraining student from current weights

This addresses covariate shift - the key limitation of behavioral cloning.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from scripts.train_parallel import VelocityRacingEnv, create_simple_track
from scripts.train_vision_student import VisionStudentNetV2, DemoDataset, train_student


class DAggerCollector:
    """Collects DAgger data by running student and querying teacher."""

    def __init__(
        self,
        student_path: str,
        teacher_path: str,
        output_dir: str,
        device: str = "cuda",
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)

        # Load student (vision-based)
        checkpoint = torch.load(student_path, map_location=device, weights_only=False)
        self.num_frames = checkpoint.get("num_frames", 4)
        self.student = VisionStudentNetV2(
            action_dim=4,
            hidden_dims=(256, 256),
            device=device,
            num_frames=self.num_frames,
        ).to(device)
        self.student.load_state_dict(checkpoint["model_state_dict"])
        self.student.eval()
        print(f"Loaded student from {student_path} (num_frames={self.num_frames})")

        # Load teacher (state-based)
        self.teacher = PPO.load(teacher_path)
        print(f"Loaded teacher from {teacher_path}")

        self.frame_count = 0

    def collect_episode(self, env, max_steps: int = 1000, beta: float = 0.0):
        """
        Collect one episode of DAgger data.

        Args:
            env: Racing environment
            max_steps: Maximum steps per episode
            beta: Probability of using teacher action (0 = pure student, 1 = pure teacher)
                  DAgger typically starts with beta=1 and decays to 0

        Returns:
            List of (image_path, teacher_action, gates_passed) tuples
        """
        obs, _ = env.reset()
        done = False
        step = 0
        episode_data = []
        frame_buffer = []

        while not done and step < max_steps:
            # Get camera image
            rgb = env.get_camera_image()
            rgb_norm = rgb.astype(np.float32) / 255.0

            # Update frame buffer
            frame_buffer.append(rgb_norm)
            if len(frame_buffer) > self.num_frames:
                frame_buffer.pop(0)

            # Pad if not enough frames yet
            if len(frame_buffer) < self.num_frames:
                padded = [frame_buffer[0]] * (self.num_frames - len(frame_buffer)) + frame_buffer
            else:
                padded = frame_buffer

            # Stack frames for student input
            stacked = np.concatenate(padded, axis=-1)
            img_tensor = torch.from_numpy(stacked).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Get student action
            with torch.no_grad():
                student_action = self.student(img_tensor).squeeze().cpu().numpy()

            # Get teacher action (using ground truth state)
            teacher_action, _ = self.teacher.predict(obs, deterministic=True)
            teacher_action = teacher_action.squeeze()

            # Decide which action to execute (DAgger mixing)
            if np.random.random() < beta:
                exec_action = teacher_action
            else:
                exec_action = student_action

            # Save data: student's observation, teacher's action
            img_path = self.output_dir / "images" / f"frame_{self.frame_count:06d}.npy"
            np.save(img_path, rgb)
            episode_data.append({
                "image_path": str(img_path),
                "action": teacher_action.tolist(),  # Always store teacher action
                "student_action": student_action.tolist(),  # For analysis
                "frame_id": self.frame_count,
            })
            self.frame_count += 1

            # Step environment
            obs, reward, terminated, truncated, info = env.step(exec_action.reshape(1, -1))
            done = terminated or truncated
            step += 1

        gates_passed = info.get("gates_passed", 0)
        return episode_data, gates_passed

    def collect(
        self,
        num_episodes: int = 50,
        max_steps: int = 1000,
        beta: float = 0.5,
        num_gates: int = 5,
    ):
        """
        Collect DAgger dataset.

        Args:
            num_episodes: Number of episodes to collect
            max_steps: Max steps per episode
            beta: Teacher mixing probability
            num_gates: Number of gates in track
        """
        print("=" * 70)
        print("DAGGER DATA COLLECTION")
        print("=" * 70)
        print(f"Episodes: {num_episodes}")
        print(f"Beta (teacher prob): {beta}")
        print()

        # Create environment
        track = create_simple_track(num_gates=num_gates, radius=1.5)

        # Import here to avoid pybullet dep at top level
        from scripts.collect_teacher_demos import CameraRacingEnv

        env = CameraRacingEnv(
            track=track,
            image_size=(64, 48),
            ctrl_freq=48,
            pyb_freq=240,
            gui=False,
            gate_tolerance=0.8,
            max_steps=max_steps,
        )

        all_data = []
        total_gates = 0

        for ep in range(num_episodes):
            ep_data, gates = self.collect_episode(env, max_steps, beta)
            all_data.extend(ep_data)
            total_gates += gates

            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{num_episodes}: {gates}/{num_gates} gates, "
                      f"total frames: {len(all_data)}")

        env.close()

        # Save metadata
        with open(self.output_dir / "demos.json", "w") as f:
            # Add episode info
            for i, d in enumerate(all_data):
                d["episode"] = i // max_steps
            json.dump(all_data, f)

        print()
        print(f"Collected {len(all_data)} frames from {num_episodes} episodes")
        print(f"Average gates: {total_gates / num_episodes:.1f}/{num_gates}")
        print(f"Saved to {self.output_dir}")

        return all_data


def merge_datasets(original_dir: str, dagger_dir: str, output_dir: str):
    """Merge original BC dataset with new DAgger data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load both datasets
    with open(Path(original_dir) / "demos.json") as f:
        original_data = json.load(f)
    with open(Path(dagger_dir) / "demos.json") as f:
        dagger_data = json.load(f)

    # Merge (DAgger data gets higher episode numbers)
    max_ep = max(d.get("episode", 0) for d in original_data) + 1
    for d in dagger_data:
        d["episode"] = d.get("episode", 0) + max_ep

    merged = original_data + dagger_data

    # Save merged dataset
    with open(output_path / "demos.json", "w") as f:
        json.dump(merged, f)

    print(f"Merged {len(original_data)} + {len(dagger_data)} = {len(merged)} frames")
    print(f"Saved to {output_path}")

    return merged


def run_dagger_iteration(
    student_path: str,
    teacher_path: str,
    original_data_dir: str,
    output_dir: str,
    iteration: int,
    num_episodes: int = 50,
    beta: float = 0.5,
    epochs: int = 10,
    device: str = "cuda",
):
    """
    Run one iteration of DAgger.

    1. Collect new data with current student
    2. Merge with existing data
    3. Retrain student
    """
    iter_dir = Path(output_dir) / f"iter_{iteration:02d}"
    dagger_data_dir = iter_dir / "dagger_data"
    merged_data_dir = iter_dir / "merged_data"
    model_dir = iter_dir / "model"

    print(f"\n{'='*70}")
    print(f"DAGGER ITERATION {iteration}")
    print(f"{'='*70}\n")

    # Step 1: Collect DAgger data
    collector = DAggerCollector(
        student_path=student_path,
        teacher_path=teacher_path,
        output_dir=str(dagger_data_dir),
        device=device,
    )
    collector.collect(num_episodes=num_episodes, beta=beta)

    # Step 2: Merge datasets
    # For iteration 0, use original data
    # For iteration > 0, use previous iteration's merged data
    if iteration == 0:
        prev_data_dir = original_data_dir
    else:
        prev_data_dir = str(Path(output_dir) / f"iter_{iteration-1:02d}" / "merged_data")

    # Create merged dataset (symlink images, merge JSON)
    merged_data_dir.mkdir(parents=True, exist_ok=True)
    (merged_data_dir / "images").mkdir(exist_ok=True)

    # Copy/link image directories
    import shutil
    # For simplicity, we'll just reference paths in the JSON
    merge_datasets(prev_data_dir, str(dagger_data_dir), str(merged_data_dir))

    # Step 3: Retrain student
    model_dir.mkdir(parents=True, exist_ok=True)
    new_model_path = str(model_dir / "best_model.pt")

    print(f"\nRetraining student for {epochs} epochs...")
    train_student(
        demos_path=str(merged_data_dir),
        output_path=new_model_path,
        epochs=epochs,
        batch_size=2048,
        lr=3e-4,
        device=device,
        num_frames=4,
    )

    return new_model_path


def main():
    parser = argparse.ArgumentParser(description="Run DAgger for vision-based drone racing")
    parser.add_argument("--student", default="models/vision_student/best_model.pt",
                        help="Path to student model")
    parser.add_argument("--teacher", default="models/curriculum_final.zip",
                        help="Path to teacher model")
    parser.add_argument("--data", default="data/dart_demos",
                        help="Path to original BC dataset")
    parser.add_argument("--output", default="data/dagger",
                        help="Output directory for DAgger iterations")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of DAgger iterations")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per iteration")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per iteration")
    parser.add_argument("--beta-start", type=float, default=0.5,
                        help="Initial teacher mixing probability")
    parser.add_argument("--beta-decay", type=float, default=0.5,
                        help="Beta decay per iteration")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda, cpu)")
    args = parser.parse_args()

    student_path = args.student
    beta = args.beta_start

    for i in range(args.iterations):
        print(f"\n{'#'*70}")
        print(f"# DAGGER ITERATION {i+1}/{args.iterations} (beta={beta:.2f})")
        print(f"{'#'*70}")

        new_student_path = run_dagger_iteration(
            student_path=student_path,
            teacher_path=args.teacher,
            original_data_dir=args.data,
            output_dir=args.output,
            iteration=i,
            num_episodes=args.episodes,
            beta=beta,
            epochs=args.epochs,
            device=args.device,
        )

        # Update for next iteration
        student_path = new_student_path
        beta *= args.beta_decay

        # Evaluate
        print(f"\nEvaluating iteration {i} model...")
        import subprocess
        result = subprocess.run(
            ["python", "-u", "scripts/eval_vision_student.py",
             "--model", new_student_path,
             "--episodes", "10",
             "--device", args.device],
            capture_output=True, text=True
        )
        print(result.stdout)

    print(f"\n{'='*70}")
    print("DAGGER COMPLETE")
    print(f"{'='*70}")
    print(f"Final model: {student_path}")


if __name__ == "__main__":
    main()
