#!/usr/bin/env python3
"""
Collect teacher demonstrations for vision-based imitation learning.

Runs the trained teacher policy with camera enabled and saves:
- Camera RGB images (64x48)
- Teacher actions (velocity commands)
- Ground truth state (for debugging)

This data is used to train a vision-based student policy via behavioral cloning.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC, PPO
from scripts.train_parallel import VelocityRacingEnv, create_simple_track
from src.vision.gate_net import create_gatenet


class TeacherDemoCollector:
    """Collects demonstrations from a trained teacher policy."""

    def __init__(
        self,
        teacher_path: str,
        output_dir: str,
        image_size: tuple = (64, 48),
        gatenet_path: str = None,
    ):
        self.teacher_path = teacher_path
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.gatenet_path = gatenet_path

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)

        # Load teacher
        self.teacher = self._load_teacher()

        # Load GateNet if provided (for feature extraction)
        self.gatenet = None
        if gatenet_path:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            self.gatenet = create_gatenet(gatenet_path, device=device)
            self.gatenet.eval()
            self.device = device

        # Collection stats
        self.total_frames = 0
        self.total_episodes = 0
        self.total_gates = 0

    def _load_teacher(self):
        """Load the trained teacher policy."""
        try:
            return PPO.load(self.teacher_path)
        except Exception:
            return SAC.load(self.teacher_path)

    def collect(
        self,
        num_episodes: int = 100,
        max_steps: int = 1000,
        num_gates: int = 5,
        gate_tolerance: float = 0.8,
        save_features: bool = True,
    ):
        """
        Collect demonstrations from the teacher.

        Args:
            num_episodes: Number of episodes to collect
            max_steps: Max steps per episode
            num_gates: Number of gates in track
            gate_tolerance: Gate pass tolerance
            save_features: Also save GateNet features (if gatenet loaded)
        """
        print("=" * 70)
        print("COLLECTING TEACHER DEMONSTRATIONS")
        print("=" * 70)
        print(f"Teacher: {self.teacher_path}")
        print(f"Output: {self.output_dir}")
        print(f"Episodes: {num_episodes}")
        print()

        # Create environment with camera
        track = create_simple_track(num_gates=num_gates, radius=1.5)
        env = CameraRacingEnv(
            track=track,
            image_size=self.image_size,
            ctrl_freq=48,
            pyb_freq=240,
            gui=False,
            gate_tolerance=gate_tolerance,
            max_steps=max_steps,
        )

        # Demonstration storage
        demos = []

        for ep in range(num_episodes):
            obs, _ = env.reset()
            episode_demos = []
            done = False

            while not done:
                # Get camera image
                rgb = env.get_camera_image()

                # Get teacher action
                action, _ = self.teacher.predict(obs, deterministic=True)

                # Get ground truth state for debugging
                state = env._getDroneStateVector(0)
                drone_pos = state[0:3].copy()
                drone_vel = state[10:13].copy()

                # Extract GateNet features if available
                features = None
                if self.gatenet is not None and save_features:
                    features = self._extract_features(rgb)

                # Save demonstration
                frame_data = {
                    "frame_id": self.total_frames,
                    "episode": ep,
                    "action": action.tolist(),
                    "drone_pos": drone_pos.tolist(),
                    "drone_vel": drone_vel.tolist(),
                    "current_gate": env.current_gate,
                    "gates_passed": env.gates_passed,
                }

                # Save image
                img_path = self.output_dir / "images" / f"frame_{self.total_frames:06d}.npy"
                np.save(img_path, rgb)
                frame_data["image_path"] = str(img_path)

                # Save features if extracted
                if features is not None:
                    feat_path = self.output_dir / "images" / f"feat_{self.total_frames:06d}.npy"
                    np.save(feat_path, features)
                    frame_data["feature_path"] = str(feat_path)

                episode_demos.append(frame_data)
                self.total_frames += 1

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Episode complete
            gates_passed = info.get("gates_passed", 0)
            self.total_gates += gates_passed
            self.total_episodes += 1

            # Only keep successful episodes (passed at least 1 gate)
            if gates_passed > 0:
                demos.extend(episode_demos)

            if (ep + 1) % 10 == 0:
                avg_gates = self.total_gates / self.total_episodes
                print(f"Episode {ep+1}/{num_episodes}: {gates_passed}/{num_gates} gates, "
                      f"avg={avg_gates:.2f}, frames={len(demos)}")

        env.close()

        # Save demonstrations metadata
        metadata = {
            "teacher_path": self.teacher_path,
            "num_episodes": num_episodes,
            "num_frames": len(demos),
            "num_gates": num_gates,
            "image_size": list(self.image_size),
            "timestamp": datetime.now().isoformat(),
            "avg_gates_passed": self.total_gates / self.total_episodes,
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save frame data
        with open(self.output_dir / "demos.json", "w") as f:
            json.dump(demos, f)

        print()
        print("=" * 70)
        print("COLLECTION COMPLETE")
        print("=" * 70)
        print(f"Total frames: {len(demos)}")
        print(f"Total episodes: {self.total_episodes}")
        print(f"Average gates: {self.total_gates / self.total_episodes:.2f}")
        print(f"Saved to: {self.output_dir}")

        return demos

    def _extract_features(self, rgb: np.ndarray) -> np.ndarray:
        """Extract features using GateNet encoder."""
        # Prepare input: (H, W, 3) -> (1, 3, H, W)
        img = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        img = img / 255.0
        img = img.to(self.device)

        with torch.no_grad():
            # Get encoder features (before final layer)
            features = self.gatenet.get_encoder_features(img)

        return features.squeeze().cpu().numpy()


class CameraRacingEnv(VelocityRacingEnv):
    """Racing environment with camera capture."""

    def __init__(self, image_size=(64, 48), **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self._setup_camera()

    def _setup_camera(self):
        """Setup camera parameters."""
        import pybullet as p

        width, height = self.image_size
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=width / height,
            nearVal=0.01,
            farVal=100.0,
        )

    def get_camera_image(self) -> np.ndarray:
        """Capture RGB image from drone camera."""
        import pybullet as p

        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]

        # Compute camera orientation
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        forward = rot_matrix @ np.array([1, 0, 0])
        up = rot_matrix @ np.array([0, 0, 1])

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=pos,
            cameraTargetPosition=pos + forward,
            cameraUpVector=up,
        )

        width, height = self.image_size
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        # Convert to numpy (H, W, 4) -> (H, W, 3)
        rgb = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        return rgb


def main():
    parser = argparse.ArgumentParser(description="Collect teacher demonstrations")
    parser.add_argument("--teacher", default="models/curriculum_final.zip",
                        help="Path to trained teacher model")
    parser.add_argument("--output", default="data/teacher_demos",
                        help="Output directory for demonstrations")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per episode")
    parser.add_argument("--gates", type=int, default=5,
                        help="Number of gates")
    parser.add_argument("--gatenet", default="models/gate_net/best_model.pt",
                        help="Path to GateNet for feature extraction")
    parser.add_argument("--no-features", action="store_true",
                        help="Don't save GateNet features")
    args = parser.parse_args()

    collector = TeacherDemoCollector(
        teacher_path=args.teacher,
        output_dir=args.output,
        gatenet_path=args.gatenet if not args.no_features else None,
    )

    collector.collect(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        num_gates=args.gates,
        save_features=not args.no_features,
    )


if __name__ == "__main__":
    main()
