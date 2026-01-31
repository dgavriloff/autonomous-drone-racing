"""
Data collection module for training the vision pipeline.

Collects synchronized camera images and ground truth gate positions
from the gym-pybullet-drones simulation for training GateNet.
"""

import numpy as np
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time

import pybullet as p


@dataclass
class GateInfo:
    """Information about a racing gate."""
    id: int
    position: np.ndarray  # 3D world position
    orientation: np.ndarray  # quaternion (x, y, z, w)
    width: float
    height: float
    corners_world: np.ndarray  # 4x3 corner positions in world frame


@dataclass
class FrameData:
    """Data for a single collected frame."""
    frame_id: int
    timestamp: float
    # Drone state
    drone_position: np.ndarray
    drone_orientation: np.ndarray  # quaternion
    drone_velocity: np.ndarray
    # Camera images (saved as paths)
    rgb_path: str
    depth_path: Optional[str]
    seg_path: Optional[str]
    # Gate information in drone/camera frame
    visible_gates: List[int]
    gate_corners_image: Dict[int, np.ndarray]  # gate_id -> 4x2 image coords
    gate_corners_camera: Dict[int, np.ndarray]  # gate_id -> 4x3 camera coords
    gate_masks: Dict[int, np.ndarray]  # gate_id -> binary mask


class DataCollector:
    """
    Collects training data from drone racing simulations.

    Captures:
    - RGB images (64x48) from drone camera
    - Depth images
    - Segmentation masks
    - Ground truth gate positions and corners
    - Gate visibility masks
    """

    def __init__(
        self,
        output_dir: str,
        image_size: Tuple[int, int] = (64, 48),
        save_depth: bool = True,
        save_segmentation: bool = True,
        camera_fov: float = 60.0,
        camera_near: float = 0.01,
        camera_far: float = 100.0,
    ):
        """
        Initialize data collector.

        Args:
            output_dir: Directory to save collected data
            image_size: (width, height) of captured images
            save_depth: Whether to save depth images
            save_segmentation: Whether to save segmentation masks
            camera_fov: Camera field of view in degrees
            camera_near: Near clipping plane
            camera_far: Far clipping plane
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.save_depth = save_depth
        self.save_segmentation = save_segmentation
        self.camera_fov = camera_fov
        self.camera_near = camera_near
        self.camera_far = camera_far

        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        # Gate registry
        self.gates: Dict[int, GateInfo] = {}

        # Collection state
        self.frame_count = 0
        self.frames: List[FrameData] = []

        # Camera intrinsics (computed from FOV and image size)
        self._compute_camera_intrinsics()

    def _compute_camera_intrinsics(self):
        """Compute camera intrinsic matrix from FOV and image size."""
        width, height = self.image_size
        fov_rad = np.radians(self.camera_fov)

        # Focal length in pixels
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Square pixels assumed

        # Principal point at image center
        cx = width / 2
        cy = height / 2

        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def register_gate(
        self,
        gate_id: int,
        position: np.ndarray,
        orientation: np.ndarray,
        width: float = 1.0,
        height: float = 1.0,
    ):
        """
        Register a gate in the scene for tracking.

        Args:
            gate_id: Unique gate identifier
            position: 3D position in world frame
            orientation: Quaternion (x, y, z, w) in world frame
            width: Gate width in meters
            height: Gate height in meters
        """
        # Compute corner positions in world frame
        corners_local = np.array([
            [-width/2, -height/2, 0],
            [width/2, -height/2, 0],
            [width/2, height/2, 0],
            [-width/2, height/2, 0],
        ])

        # Convert quaternion to rotation matrix
        rot_matrix = self._quat_to_rotation_matrix(orientation)

        # Transform corners to world frame
        corners_world = (rot_matrix @ corners_local.T).T + position

        self.gates[gate_id] = GateInfo(
            id=gate_id,
            position=np.array(position),
            orientation=np.array(orientation),
            width=width,
            height=height,
            corners_world=corners_world,
        )

    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
        x, y, z, w = quat

        # Normalize quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])

    def capture_frame(
        self,
        drone_position: np.ndarray,
        drone_orientation: np.ndarray,
        drone_velocity: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        seg_image: Optional[np.ndarray] = None,
        physics_client: Optional[int] = None,
    ) -> FrameData:
        """
        Capture and save a single frame of training data.

        Args:
            drone_position: Drone position in world frame
            drone_orientation: Drone orientation as quaternion (x, y, z, w)
            drone_velocity: Drone velocity in world frame
            rgb_image: RGB image from drone camera (H, W, 3 or 4)
            depth_image: Optional depth image
            seg_image: Optional segmentation image
            physics_client: Optional PyBullet physics client for raycasting

        Returns:
            FrameData object with all captured information
        """
        frame_id = self.frame_count
        timestamp = time.time()

        # Save RGB image
        rgb_path = self.images_dir / f"rgb_{frame_id:06d}.npy"
        np.save(rgb_path, rgb_image)

        # Save depth if provided
        depth_path = None
        if depth_image is not None and self.save_depth:
            depth_path = self.images_dir / f"depth_{frame_id:06d}.npy"
            np.save(depth_path, depth_image)

        # Save segmentation if provided
        seg_path = None
        if seg_image is not None and self.save_segmentation:
            seg_path = self.images_dir / f"seg_{frame_id:06d}.npy"
            np.save(seg_path, seg_image)

        # Compute camera pose (world to camera transform)
        drone_rot = self._quat_to_rotation_matrix(drone_orientation)

        # Camera is mounted looking forward (x-axis in drone frame)
        # Standard camera convention: z forward, x right, y down
        # Drone convention: x forward, y left, z up
        # Transform: camera_z = drone_x, camera_x = -drone_y, camera_y = -drone_z
        R_drone_to_cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
        ])

        R_world_to_cam = R_drone_to_cam @ drone_rot.T
        t_world_to_cam = -R_world_to_cam @ drone_position

        # Process each gate
        visible_gates = []
        gate_corners_image = {}
        gate_corners_camera = {}
        gate_masks = {}

        for gate_id, gate in self.gates.items():
            # Transform gate corners to camera frame
            corners_camera = (R_world_to_cam @ gate.corners_world.T).T + t_world_to_cam

            # Check visibility (all corners in front of camera)
            if np.all(corners_camera[:, 2] > self.camera_near):
                # Project to image coordinates
                corners_image = self._project_to_image(corners_camera)

                # Check if any corner is in image bounds
                w, h = self.image_size
                in_bounds = (
                    (corners_image[:, 0] >= 0) & (corners_image[:, 0] < w) &
                    (corners_image[:, 1] >= 0) & (corners_image[:, 1] < h)
                )

                if np.any(in_bounds):
                    visible_gates.append(gate_id)
                    gate_corners_image[gate_id] = corners_image
                    gate_corners_camera[gate_id] = corners_camera

                    # Generate gate mask
                    mask = self._generate_gate_mask(corners_image)
                    gate_masks[gate_id] = mask

                    # Save mask
                    mask_path = self.masks_dir / f"gate_{gate_id}_frame_{frame_id:06d}.npy"
                    np.save(mask_path, mask)

        # Create frame data
        frame_data = FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            drone_position=drone_position.copy(),
            drone_orientation=drone_orientation.copy(),
            drone_velocity=drone_velocity.copy(),
            rgb_path=str(rgb_path),
            depth_path=str(depth_path) if depth_path else None,
            seg_path=str(seg_path) if seg_path else None,
            visible_gates=visible_gates,
            gate_corners_image={k: v.tolist() for k, v in gate_corners_image.items()},
            gate_corners_camera={k: v.tolist() for k, v in gate_corners_camera.items()},
            gate_masks={k: str(self.masks_dir / f"gate_{k}_frame_{frame_id:06d}.npy")
                       for k in gate_masks.keys()},
        )

        self.frames.append(frame_data)
        self.frame_count += 1

        return frame_data

    def _project_to_image(self, points_camera: np.ndarray) -> np.ndarray:
        """
        Project 3D points in camera frame to 2D image coordinates.

        Args:
            points_camera: Nx3 points in camera frame

        Returns:
            Nx2 image coordinates
        """
        # Perspective projection
        x = points_camera[:, 0] / points_camera[:, 2]
        y = points_camera[:, 1] / points_camera[:, 2]

        # Apply camera intrinsics
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        return np.stack([u, v], axis=1)

    def _generate_gate_mask(self, corners_image: np.ndarray) -> np.ndarray:
        """
        Generate binary mask for gate from projected corners.

        Args:
            corners_image: 4x2 corner coordinates in image

        Returns:
            Binary mask of shape (H, W)
        """
        import cv2

        w, h = self.image_size
        mask = np.zeros((h, w), dtype=np.uint8)

        # Convert corners to integer pixel coordinates
        corners_int = corners_image.astype(np.int32)

        # Fill polygon
        cv2.fillPoly(mask, [corners_int], 255)

        return mask

    def save_metadata(self):
        """Save collection metadata and frame information."""
        metadata = {
            "num_frames": self.frame_count,
            "image_size": self.image_size,
            "camera_fov": self.camera_fov,
            "camera_matrix": self.camera_matrix.tolist(),
            "gates": {
                str(k): {
                    "id": v.id,
                    "position": v.position.tolist(),
                    "orientation": v.orientation.tolist(),
                    "width": v.width,
                    "height": v.height,
                    "corners_world": v.corners_world.tolist(),
                }
                for k, v in self.gates.items()
            },
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save frame data
        frames_data = []
        for frame in self.frames:
            frame_dict = {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "drone_position": frame.drone_position.tolist(),
                "drone_orientation": frame.drone_orientation.tolist(),
                "drone_velocity": frame.drone_velocity.tolist(),
                "rgb_path": frame.rgb_path,
                "depth_path": frame.depth_path,
                "seg_path": frame.seg_path,
                "visible_gates": frame.visible_gates,
                "gate_corners_image": frame.gate_corners_image,
                "gate_corners_camera": frame.gate_corners_camera,
                "gate_masks": frame.gate_masks,
            }
            frames_data.append(frame_dict)

        with open(self.output_dir / "frames.json", "w") as f:
            json.dump(frames_data, f, indent=2)

        print(f"Saved metadata and {self.frame_count} frames to {self.output_dir}")

    @classmethod
    def load(cls, data_dir: str) -> Tuple["DataCollector", List[FrameData]]:
        """
        Load previously collected data.

        Args:
            data_dir: Directory containing collected data

        Returns:
            Tuple of (DataCollector instance, list of FrameData)
        """
        data_dir = Path(data_dir)

        # Load metadata
        with open(data_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        collector = cls(
            output_dir=str(data_dir),
            image_size=tuple(metadata["image_size"]),
            camera_fov=metadata["camera_fov"],
        )
        collector.camera_matrix = np.array(metadata["camera_matrix"])

        # Load gates
        for gate_id, gate_data in metadata["gates"].items():
            collector.gates[int(gate_id)] = GateInfo(
                id=gate_data["id"],
                position=np.array(gate_data["position"]),
                orientation=np.array(gate_data["orientation"]),
                width=gate_data["width"],
                height=gate_data["height"],
                corners_world=np.array(gate_data["corners_world"]),
            )

        # Load frames
        with open(data_dir / "frames.json", "r") as f:
            frames_data = json.load(f)

        frames = []
        for fd in frames_data:
            frame = FrameData(
                frame_id=fd["frame_id"],
                timestamp=fd["timestamp"],
                drone_position=np.array(fd["drone_position"]),
                drone_orientation=np.array(fd["drone_orientation"]),
                drone_velocity=np.array(fd["drone_velocity"]),
                rgb_path=fd["rgb_path"],
                depth_path=fd["depth_path"],
                seg_path=fd["seg_path"],
                visible_gates=fd["visible_gates"],
                gate_corners_image=fd["gate_corners_image"],
                gate_corners_camera=fd["gate_corners_camera"],
                gate_masks=fd["gate_masks"],
            )
            frames.append(frame)

        collector.frames = frames
        collector.frame_count = len(frames)

        return collector, frames


class GateDataset:
    """
    PyTorch-compatible dataset for gate segmentation training.

    Loads RGB images and corresponding gate masks for training GateNet.
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        augment: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing collected data
            transform: Optional transforms to apply
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment

        # Load metadata and frames
        self.collector, self.frames = DataCollector.load(data_dir)

        # Filter to frames with visible gates
        self.valid_frames = [
            f for f in self.frames if len(f.visible_gates) > 0
        ]

        print(f"Loaded {len(self.valid_frames)} valid frames "
              f"({len(self.frames)} total)")

    def __len__(self) -> int:
        return len(self.valid_frames)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (RGB image, combined gate mask)
        """
        frame = self.valid_frames[idx]

        # Load RGB image
        rgb = np.load(frame.rgb_path)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]  # Remove alpha channel

        # Combine all gate masks
        h, w = rgb.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)

        for gate_id in frame.visible_gates:
            if gate_id in frame.gate_masks:
                mask_path = frame.gate_masks[gate_id]
                if isinstance(mask_path, str):
                    mask = np.load(mask_path)
                    combined_mask = np.maximum(combined_mask, mask.astype(np.float32) / 255.0)

        # Apply augmentation if enabled
        if self.augment:
            rgb, combined_mask = self._augment(rgb, combined_mask)

        # Normalize RGB to [0, 1]
        rgb = rgb.astype(np.float32) / 255.0

        # Apply transforms if provided
        if self.transform:
            rgb = self.transform(rgb)

        return rgb, combined_mask

    def _augment(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            rgb = np.fliplr(rgb).copy()
            mask = np.fliplr(mask).copy()

        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        rgb = np.clip(rgb * brightness, 0, 255).astype(np.uint8)

        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 5, rgb.shape).astype(np.float32)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return rgb, mask


def collect_from_simulation(
    output_dir: str,
    num_frames: int = 50000,
    track_gates: List[Dict[str, Any]] = None,
) -> DataCollector:
    """
    Collect training data from gym-pybullet-drones simulation.

    This function runs a simulation with a PID controller following
    a racing trajectory and collects camera images with ground truth
    gate positions.

    Args:
        output_dir: Directory to save collected data
        num_frames: Target number of frames to collect
        track_gates: List of gate definitions with position, orientation, size

    Returns:
        DataCollector with collected data
    """
    from gym_pybullet_drones.envs import CtrlAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.control import DSLPIDControl

    # Default track if none provided
    if track_gates is None:
        track_gates = [
            {"position": [2, 0, 1], "orientation": [0, 0, 0, 1], "width": 1.0, "height": 1.0},
            {"position": [4, 2, 1.2], "orientation": [0, 0, 0.38, 0.92], "width": 1.0, "height": 1.0},
            {"position": [4, 4, 1], "orientation": [0, 0, 0.71, 0.71], "width": 1.0, "height": 1.0},
            {"position": [2, 4, 0.8], "orientation": [0, 0, 0.92, 0.38], "width": 1.0, "height": 1.0},
            {"position": [0, 2, 1], "orientation": [0, 0, 1, 0], "width": 1.0, "height": 1.0},
        ]

    # Initialize collector
    collector = DataCollector(output_dir=output_dir)

    # Register gates
    for i, gate in enumerate(track_gates):
        collector.register_gate(
            gate_id=i,
            position=np.array(gate["position"]),
            orientation=np.array(gate["orientation"]),
            width=gate.get("width", 1.0),
            height=gate.get("height", 1.0),
        )

    # Create environment with vision
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        physics=Physics.PYB,
        gui=False,
        record=False,
        vision_attributes=True,
    )

    # Initialize controller
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

    # Generate waypoints from gates
    waypoints = np.array([g["position"] for g in track_gates])

    print(f"Starting data collection: {num_frames} frames from {len(track_gates)} gates")

    obs, info = env.reset()
    collected = 0
    waypoint_idx = 0

    while collected < num_frames:
        # Get current state
        state = env._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        vel = state[10:13]

        # Get camera images
        rgb, depth, seg = env._getDroneImages(0)

        # Capture frame
        collector.capture_frame(
            drone_position=pos,
            drone_orientation=quat,
            drone_velocity=vel,
            rgb_image=rgb,
            depth_image=depth,
            seg_image=seg,
        )
        collected += 1

        if collected % 1000 == 0:
            print(f"Collected {collected}/{num_frames} frames")

        # Compute control
        target_pos = waypoints[waypoint_idx]
        target_vel = np.zeros(3)
        target_rpy = np.zeros(3)

        action, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
        )

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action.reshape(1, 4))

        # Check if reached waypoint
        if np.linalg.norm(pos - target_pos) < 0.3:
            waypoint_idx = (waypoint_idx + 1) % len(waypoints)

        # Reset if crashed
        if terminated or truncated:
            obs, info = env.reset()
            waypoint_idx = 0

    env.close()

    # Save metadata
    collector.save_metadata()

    print(f"Data collection complete: {collector.frame_count} frames saved")
    return collector


if __name__ == "__main__":
    # Test data collection
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = collect_from_simulation(
            output_dir=tmpdir,
            num_frames=100,  # Small test
        )

        print(f"\nTest complete!")
        print(f"Frames collected: {collector.frame_count}")
        print(f"Gates registered: {len(collector.gates)}")
