#!/usr/bin/env python3
"""
Test the vision pipeline end-to-end.

Creates a VisionRacingEnv that uses:
- Camera images from PyBullet
- GateNet for segmentation
- QuAdGate for corner detection
- PoseEstimator for gate pose
- Optionally EKF for state fusion

Tests whether the trained SAC policy can complete gates using
vision-based state estimation instead of ground truth.
"""

import sys
from pathlib import Path
import numpy as np
import time
import torch
import pybullet as p

sys.path.insert(0, str(Path(__file__).parent.parent))

from gymnasium import spaces
from stable_baselines3 import SAC, PPO

from scripts.train_parallel import VelocityRacingEnv, create_simple_track
from src.vision.gate_net import GateNet, create_gatenet
from src.vision.quad_gate import QuAdGate, GateTracker
from src.vision.pose_estimator import PoseEstimator
from src.envs.high_freq_racing import TrackConfig, GateConfig


class VisionRacingEnv(VelocityRacingEnv):
    """
    Racing environment with vision-based state estimation.

    Instead of using ground truth state from PyBullet, this environment:
    1. Captures camera images from drone perspective
    2. Runs GateNet to get gate segmentation mask
    3. Runs QuAdGate to detect gate corners
    4. Runs PoseEstimator to get gate pose in camera frame
    5. Computes observation from vision-based estimates

    The observation space is identical to VelocityRacingEnv:
    [pos(3), vel(3), euler(3), ang_vel(3), to_gate_dir(3), dist] = 16 dims

    But pos, vel, euler come from vision + dead reckoning instead of ground truth.
    """

    def __init__(
        self,
        track: TrackConfig,
        gatenet_path: str = "models/gate_net/best_model.pt",
        ctrl_freq: int = 48,
        pyb_freq: int = 240,
        gui: bool = False,
        gate_tolerance: float = 0.5,
        max_steps: int = 1500,
        image_size: tuple = (64, 48),
        camera_fov: float = 60.0,
        use_ground_truth_fallback: bool = True,
        debug_vision: bool = False,
        **kwargs,
    ):
        """
        Initialize vision-based racing environment.

        Args:
            track: Track configuration
            gatenet_path: Path to trained GateNet model
            ctrl_freq: Control frequency in Hz
            pyb_freq: Physics frequency in Hz
            gui: Show GUI
            gate_tolerance: Gate pass tolerance in meters
            max_steps: Max steps per episode
            image_size: Camera image size (width, height)
            camera_fov: Camera field of view in degrees
            use_ground_truth_fallback: Fall back to ground truth if vision fails
            debug_vision: Print debug info about vision pipeline
        """
        # Initialize base environment
        super().__init__(
            track=track,
            ctrl_freq=ctrl_freq,
            pyb_freq=pyb_freq,
            gui=gui,
            gate_tolerance=gate_tolerance,
            max_steps=max_steps,
            **kwargs,
        )

        self.image_size = image_size
        self.camera_fov = camera_fov
        self.use_ground_truth_fallback = use_ground_truth_fallback
        self.debug_vision = debug_vision

        # Load vision components
        self._init_vision_pipeline(gatenet_path)

        # State estimation (dead reckoning between vision updates)
        self.estimated_position = None
        self.estimated_velocity = None
        self.estimated_orientation = None
        self.last_vision_update = 0

        # Vision statistics
        self.vision_detections = 0
        self.vision_failures = 0

    def _init_vision_pipeline(self, gatenet_path: str):
        """Initialize vision pipeline components."""
        # GateNet for segmentation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"

        self.gatenet = create_gatenet(gatenet_path, device=self.device)
        self.gatenet.eval()

        # QuAdGate for corner detection
        self.quad_gate = QuAdGate(
            min_contour_area=50,
            epsilon_factor=0.02,
        )
        self.gate_tracker = GateTracker(self.quad_gate)

        # PoseEstimator for 3D pose from corners
        self.pose_estimator = PoseEstimator(
            gate_width=1.0,  # Assumed gate dimensions
            gate_height=1.0,
            image_size=self.image_size,
            camera_fov=self.camera_fov,
        )

        # Camera parameters for PyBullet
        self._setup_camera_params()

    def _setup_camera_params(self):
        """Setup camera projection and view matrix parameters."""
        width, height = self.image_size
        fov_rad = np.radians(self.camera_fov)

        # Compute projection matrix
        aspect = width / height
        near = 0.01
        far = 100.0

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=aspect,
            nearVal=near,
            farVal=far,
        )

    def _get_camera_image(self) -> np.ndarray:
        """Capture RGB image from drone's forward-facing camera."""
        # Get drone state for camera position/orientation
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]  # (x, y, z, w) in pybullet

        # Compute camera position (at drone position)
        camera_pos = pos

        # Compute camera target (forward direction in drone frame)
        # Drone x-axis is forward
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        forward = rot_matrix @ np.array([1, 0, 0])  # x-forward
        camera_target = camera_pos + forward

        # Up vector (drone z-axis)
        up = rot_matrix @ np.array([0, 0, 1])

        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=camera_target,
            cameraUpVector=up,
        )

        # Capture image
        width, height = self.image_size
        _, _, rgb, depth, seg = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_TINY_RENDERER,  # Faster than ER_BULLET_HARDWARE_OPENGL
        )

        # Convert to numpy array (H, W, 4) -> (H, W, 3)
        rgb = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

        return rgb

    def _run_vision_pipeline(self, rgb_image: np.ndarray) -> dict:
        """
        Run the full vision pipeline on an RGB image.

        Returns:
            dict with keys:
            - 'success': bool
            - 'gate_position_camera': 3D position in camera frame
            - 'gate_distance': distance to gate
            - 'detection': GateDetection object
            - 'pose': GatePose object
        """
        result = {
            'success': False,
            'gate_position_camera': None,
            'gate_distance': None,
            'detection': None,
            'pose': None,
        }

        # 1. Run GateNet segmentation
        # Prepare input: (H, W, 3) -> (1, 3, H, W), normalized
        img_tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            mask = self.gatenet(img_tensor)

        # Convert to numpy (1, 1, H, W) -> (H, W)
        mask = mask.squeeze().cpu().numpy()

        # 2. Run QuAdGate corner detection
        detection = self.gate_tracker.update(mask)

        if detection is None:
            return result

        result['detection'] = detection

        # 3. Run PoseEstimator
        pose = self.pose_estimator.estimate_pose(detection)

        if pose is None:
            return result

        result['pose'] = pose
        result['gate_position_camera'] = pose.position
        result['gate_distance'] = pose.distance
        result['success'] = True

        return result

    def _estimate_drone_position_from_gate(
        self,
        gate_pos_camera: np.ndarray,
        drone_orientation: np.ndarray,
        gate_world_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate drone world position from gate observation.

        If we see the gate at position P_c in camera frame,
        and we know the gate world position P_w,
        then drone_position = P_w - R_world_body @ R_body_cam @ P_c

        Args:
            gate_pos_camera: Gate position in camera frame (x-right, y-down, z-forward)
            drone_orientation: Drone orientation quaternion (x, y, z, w)
            gate_world_pos: Known gate position in world frame

        Returns:
            Estimated drone position in world frame
        """
        # Camera to body frame transform
        # Camera: x-right, y-down, z-forward
        # Body: x-forward, y-left, z-up
        R_cam_to_body = np.array([
            [0, 0, 1],   # body x = cam z (forward)
            [-1, 0, 0],  # body y = -cam x (left)
            [0, -1, 0],  # body z = -cam y (up)
        ])

        # Body to world rotation
        quat = drone_orientation
        R_body_to_world = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Gate position in world frame (from drone's perspective)
        gate_pos_body = R_cam_to_body @ gate_pos_camera
        gate_pos_world_observed = R_body_to_world @ gate_pos_body

        # Drone position = known gate position - observed offset
        drone_pos = gate_world_pos - gate_pos_world_observed

        return drone_pos

    def _computeObs(self):
        """
        Compute observation using vision pipeline.

        Falls back to ground truth if vision fails and fallback is enabled.
        """
        # Get ground truth for orientation and velocity (IMU-like)
        # In real system, these would come from IMU
        state = self._getDroneStateVector(0)
        true_pos = state[0:3]
        quat = state[3:7]
        true_vel = state[10:13]
        ang_vel = state[13:16]
        euler = np.array(p.getEulerFromQuaternion(quat))

        # Get current gate info
        gate_pos = self.track.gates[self.current_gate].position

        # Try vision pipeline
        rgb = self._get_camera_image()
        vision_result = self._run_vision_pipeline(rgb)

        if vision_result['success']:
            self.vision_detections += 1

            # Estimate position from vision
            estimated_pos = self._estimate_drone_position_from_gate(
                vision_result['gate_position_camera'],
                quat,
                gate_pos,
            )

            # Use estimated position
            pos = estimated_pos

            if self.debug_vision:
                error = np.linalg.norm(estimated_pos - true_pos)
                print(f"Vision: pos_error={error:.3f}m, dist={vision_result['gate_distance']:.2f}m")

        else:
            self.vision_failures += 1

            if self.use_ground_truth_fallback:
                # Fall back to ground truth
                pos = true_pos
                if self.debug_vision:
                    print("Vision: FAILED, using ground truth fallback")
            else:
                # Use last known position or dead reckoning
                if self.estimated_position is not None:
                    # Simple dead reckoning: pos += vel * dt
                    dt = 1.0 / self.CTRL_FREQ
                    pos = self.estimated_position + true_vel * dt
                else:
                    pos = true_pos

        # Update estimated state
        self.estimated_position = pos.copy()
        self.estimated_velocity = true_vel.copy()  # Would come from IMU + integration
        self.estimated_orientation = euler.copy()

        # Compute observation (same format as VelocityRacingEnv)
        vel = true_vel  # From IMU
        to_gate = gate_pos - pos
        dist = np.linalg.norm(to_gate)
        to_gate_dir = to_gate / (dist + 1e-6)

        obs = np.concatenate([pos, vel, euler, ang_vel, to_gate_dir, [dist]])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment and vision state."""
        self.vision_detections = 0
        self.vision_failures = 0
        self.estimated_position = None
        self.estimated_velocity = None
        self.estimated_orientation = None
        self.gate_tracker.reset()

        return super().reset(seed=seed, options=options)

    def get_vision_stats(self) -> dict:
        """Get vision pipeline statistics."""
        total = self.vision_detections + self.vision_failures
        success_rate = self.vision_detections / max(1, total)
        return {
            'detections': self.vision_detections,
            'failures': self.vision_failures,
            'success_rate': success_rate,
        }


def test_vision_env(
    model_path: str = "models/curriculum_final.zip",
    gatenet_path: str = "models/gate_net/best_model.pt",
    num_episodes: int = 5,
    max_steps: int = 1500,
    gate_tolerance: float = 0.5,
    gui: bool = False,
    use_fallback: bool = True,
    debug: bool = False,
):
    """Test the vision-based environment with trained policy."""
    print("=" * 70)
    print("VISION PIPELINE END-TO-END TEST")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"GateNet: {gatenet_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Ground truth fallback: {use_fallback}")
    print()

    # Create track
    track = create_simple_track(num_gates=5, radius=1.5)

    # Create vision-based environment
    env = VisionRacingEnv(
        track=track,
        gatenet_path=gatenet_path,
        gui=gui,
        max_steps=max_steps,
        gate_tolerance=gate_tolerance,
        use_ground_truth_fallback=use_fallback,
        debug_vision=debug,
    )

    # Load policy
    try:
        model = PPO.load(model_path)
        print("Loaded PPO model")
    except Exception:
        model = SAC.load(model_path)
        print("Loaded SAC model")

    # Run episodes
    results = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            if gui:
                time.sleep(1/48)

        gates = info.get("gates_passed", 0)
        vision_stats = env.get_vision_stats()

        results.append({
            "episode": ep + 1,
            "gates": gates,
            "reward": total_reward,
            "steps": steps,
            "vision_success_rate": vision_stats['success_rate'],
        })

        print(f"Episode {ep+1}: {gates}/5 gates, "
              f"vision_rate={vision_stats['success_rate']:.1%}, "
              f"steps={steps}")

    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    avg_gates = np.mean([r["gates"] for r in results])
    max_gates = max(r["gates"] for r in results)
    full_laps = sum(1 for r in results if r["gates"] >= 5)
    avg_vision_rate = np.mean([r["vision_success_rate"] for r in results])

    print(f"Average gates: {avg_gates:.2f}/5")
    print(f"Max gates: {max_gates}/5")
    print(f"Full laps: {full_laps}/{num_episodes}")
    print(f"Avg vision success rate: {avg_vision_rate:.1%}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/curriculum_final.zip")
    parser.add_argument("--gatenet", default="models/gate_net/best_model.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable ground truth fallback when vision fails")
    parser.add_argument("--debug", action="store_true",
                        help="Print vision debug info")
    args = parser.parse_args()

    test_vision_env(
        model_path=args.model,
        gatenet_path=args.gatenet,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        gate_tolerance=args.tolerance,
        gui=args.gui,
        use_fallback=not args.no_fallback,
        debug=args.debug,
    )
