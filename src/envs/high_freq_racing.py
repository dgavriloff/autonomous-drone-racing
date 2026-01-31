"""
High-Frequency Racing Environment.

Custom racing environment with 500 Hz control frequency and
direct motor RPM action space. Supports vision-based observations
and gate-based racing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import gymnasium as gym
from gymnasium import spaces

import pybullet as p
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


@dataclass
class GateConfig:
    """Configuration for a racing gate."""
    position: np.ndarray
    orientation: np.ndarray  # quaternion (x, y, z, w)
    width: float = 1.0
    height: float = 1.0
    passed: bool = False


@dataclass
class TrackConfig:
    """Configuration for a racing track."""
    name: str
    gates: List[GateConfig] = field(default_factory=list)
    start_position: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.5]))
    start_orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))


class HighFreqRacingAviary(BaseRLAviary):
    """
    High-frequency racing environment for drone racing.

    Features:
    - 500 Hz control frequency (configurable)
    - Direct motor RPM action space
    - Gate-based racing with rewards
    - Vision support (optional)
    - State estimation compatible observations

    This environment is designed to match the requirements of the
    AI Grand Prix competition.
    """

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 2000,
        ctrl_freq: int = 500,
        gui: bool = False,
        record: bool = False,
        # Track configuration
        track: Optional[TrackConfig] = None,
        gate_tolerance: float = 0.5,
        # Observation settings
        obs_type: ObservationType = ObservationType.KIN,
        include_gate_info: bool = True,
        include_velocity_target: bool = True,
        enable_vision: bool = True,  # Enable vision for pipeline even with KIN obs
        # Reward settings
        reward_gate_passed: float = 10.0,
        reward_velocity_bonus: float = 0.1,
        reward_time_penalty: float = -0.01,
        reward_crash_penalty: float = -100.0,
        reward_orientation_penalty: float = -0.1,
        # Episode settings
        max_episode_steps: int = 1000,
        target_velocity: float = 5.0,
    ):
        """
        Initialize high-frequency racing environment.

        Args:
            drone_model: Drone model to use
            physics: Physics simulation type
            pyb_freq: PyBullet simulation frequency (Hz)
            ctrl_freq: Control/action frequency (Hz)
            gui: Whether to show GUI
            record: Whether to record video
            track: Track configuration (gates and start)
            gate_tolerance: Distance to gate center for "passed" detection
            obs_type: Observation type (KIN or RGB)
            include_gate_info: Include next gate info in observation
            include_velocity_target: Include velocity target in observation
            enable_vision: Enable vision capture even with KIN observation type
            reward_*: Reward function weights
            max_episode_steps: Maximum steps per episode
            target_velocity: Target velocity for velocity bonus
        """
        # Store parameters before parent init
        self.gate_tolerance = gate_tolerance
        self.include_gate_info = include_gate_info
        self.include_velocity_target = include_velocity_target
        self.max_episode_steps_custom = max_episode_steps
        self.target_velocity = target_velocity
        self.enable_vision = enable_vision

        # Reward weights
        self.reward_gate_passed = reward_gate_passed
        self.reward_velocity_bonus = reward_velocity_bonus
        self.reward_time_penalty = reward_time_penalty
        self.reward_crash_penalty = reward_crash_penalty
        self.reward_orientation_penalty = reward_orientation_penalty

        # Track setup
        self.track = track or self._default_track()
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.episode_steps = 0

        # Initialize parent
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs_type,
            act=ActionType.RPM,  # Direct RPM control
        )

        # Initialize vision attributes manually if enabled (for pipeline use)
        if enable_vision and not hasattr(self, 'IMG_RES'):
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ / self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros((self.NUM_DRONES, 48, 64, 4))
            self.dep = np.ones((self.NUM_DRONES, 48, 64))
            self.seg = np.zeros((self.NUM_DRONES, 48, 64))

        # Create gate visualizations (after PyBullet is initialized)
        self.gate_visual_ids = []

    def _default_track(self) -> TrackConfig:
        """Create default racing track."""
        gates = [
            GateConfig(
                position=np.array([2.0, 0.0, 1.0]),
                orientation=np.array([0, 0, 0, 1]),
            ),
            GateConfig(
                position=np.array([4.0, 2.0, 1.2]),
                orientation=np.array([0, 0, 0.38, 0.92]),
            ),
            GateConfig(
                position=np.array([4.0, 4.0, 1.0]),
                orientation=np.array([0, 0, 0.71, 0.71]),
            ),
            GateConfig(
                position=np.array([2.0, 4.0, 0.8]),
                orientation=np.array([0, 0, 0.92, 0.38]),
            ),
            GateConfig(
                position=np.array([0.0, 2.0, 1.0]),
                orientation=np.array([0, 0, 1, 0]),
            ),
        ]

        return TrackConfig(
            name="default_track",
            gates=gates,
            start_position=np.array([0.0, 0.0, 0.5]),
        )

    def _observationSpace(self) -> spaces.Box:
        """
        Define observation space.

        Base observation (12 elements):
        - Position (3)
        - Euler angles (3)
        - Velocity (3)
        - Angular velocity (3)

        + Gate info (6 elements) if include_gate_info:
        - Gate position (3)
        - Gate direction (3)

        + Velocity target (3 elements) if include_velocity_target:
        - Target velocity vector (3)

        Total: 12-21 elements depending on settings
        """
        obs_dim = 12  # Base kinematic state

        if self.include_gate_info:
            obs_dim += 6  # Gate position + direction

        if self.include_velocity_target:
            obs_dim += 3  # Velocity target

        low = -np.inf * np.ones(obs_dim)
        high = np.inf * np.ones(obs_dim)

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _actionSpace(self) -> spaces.Box:
        """
        Define action space: 4 motor RPMs.

        Actions are normalized to [0, 1] and scaled to actual RPMs internally.
        """
        return spaces.Box(
            low=np.zeros(4),
            high=np.ones(4),
            dtype=np.float32,
        )

    def _computeObs(self) -> np.ndarray:
        """Compute observation from current state."""
        # Get drone state
        state = self._getDroneStateVector(0)

        pos = state[0:3]
        quat = state[3:7]
        vel = state[10:13]
        ang_vel = state[13:16]

        # Convert quaternion to Euler
        euler = p.getEulerFromQuaternion(quat)

        # Base observation
        obs = np.concatenate([pos, euler, vel, ang_vel])

        # Add gate info
        if self.include_gate_info:
            gate = self.track.gates[self.current_gate_idx]
            gate_pos = gate.position
            gate_dir = self._get_gate_direction(gate)
            obs = np.concatenate([obs, gate_pos, gate_dir])

        # Add velocity target
        if self.include_velocity_target:
            # Target velocity towards current gate
            gate_pos = self.track.gates[self.current_gate_idx].position
            to_gate = gate_pos - pos
            to_gate_norm = to_gate / (np.linalg.norm(to_gate) + 1e-6)
            vel_target = to_gate_norm * self.target_velocity
            obs = np.concatenate([obs, vel_target])

        return obs.astype(np.float32)

    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """
        Convert normalized action to motor RPMs.

        Args:
            action: Normalized action in [0, 1]^4

        Returns:
            Motor RPMs
        """
        # Scale from [0, 1] to [0, MAX_RPM]
        return action * self.MAX_RPM

    def _computeReward(self) -> float:
        """Compute reward for current state."""
        reward = 0.0

        # Get state
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        euler = p.getEulerFromQuaternion(state[3:7])

        # Time penalty (encourage fast completion)
        reward += self.reward_time_penalty

        # Velocity bonus (reward moving fast)
        speed = np.linalg.norm(vel)
        reward += self.reward_velocity_bonus * speed

        # Orientation penalty (penalize excessive tilt)
        tilt = np.abs(euler[0]) + np.abs(euler[1])
        if tilt > 0.5:  # ~30 degrees
            reward += self.reward_orientation_penalty * tilt

        # Gate passed bonus
        gate = self.track.gates[self.current_gate_idx]
        dist_to_gate = np.linalg.norm(pos - gate.position)

        if dist_to_gate < self.gate_tolerance:
            # Check if we're on the correct side of the gate
            gate_dir = self._get_gate_direction(gate)
            to_gate = gate.position - pos
            if np.dot(to_gate, gate_dir) < 0:  # Passed through
                reward += self.reward_gate_passed
                self._advance_gate()

        return reward

    def _computeTerminated(self) -> bool:
        """Check if episode should terminate (success or failure)."""
        # Get state
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        euler = p.getEulerFromQuaternion(state[3:7])

        # Crash detection: too low or too tilted
        if pos[2] < 0.05:  # Ground contact
            return True

        if np.abs(euler[0]) > 1.2 or np.abs(euler[1]) > 1.2:  # ~70 degrees
            return True

        # Success: all gates passed
        if self.gates_passed >= len(self.track.gates):
            return True

        return False

    def _computeTruncated(self) -> bool:
        """Check if episode should truncate (timeout)."""
        return self.episode_steps >= self.max_episode_steps_custom

    def _computeInfo(self) -> Dict[str, Any]:
        """Compute info dictionary."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]

        return {
            "position": pos.copy(),
            "velocity": vel.copy(),
            "speed": np.linalg.norm(vel),
            "gates_passed": self.gates_passed,
            "current_gate": self.current_gate_idx,
            "episode_steps": self.episode_steps,
        }

    def _get_gate_direction(self, gate: GateConfig) -> np.ndarray:
        """Get unit vector pointing through gate (normal direction)."""
        # Convert quaternion to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(gate.orientation)).reshape(3, 3)
        # Gate normal is the local X axis
        return rot_matrix[:, 0]

    def _advance_gate(self):
        """Advance to next gate in sequence."""
        self.track.gates[self.current_gate_idx].passed = True
        self.gates_passed += 1
        self.current_gate_idx = (self.current_gate_idx + 1) % len(self.track.gates)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        # Reset gate tracking
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.episode_steps = 0

        for gate in self.track.gates:
            gate.passed = False

        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step the environment.

        Args:
            action: Normalized motor RPM action [0, 1]^4

        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        self.episode_steps += 1

        # Call parent step
        obs, reward_base, terminated, truncated, info = super().step(action)

        # Use our custom reward
        reward = self._computeReward()

        # Add crash penalty if terminated
        if terminated and self.gates_passed < len(self.track.gates):
            reward += self.reward_crash_penalty

        # Check our termination/truncation
        terminated = terminated or self._computeTerminated()
        truncated = truncated or self._computeTruncated()

        # Update info
        info.update(self._computeInfo())

        return obs, reward, terminated, truncated, info

    def get_state_for_control(self) -> Dict[str, np.ndarray]:
        """
        Get state in format suitable for G&CNet controller.

        Returns:
            Dict with position, velocity, orientation, angular_velocity,
            gate_position, gate_direction, velocity_target
        """
        state = self._getDroneStateVector(0)

        pos = state[0:3]
        quat_xyzw = state[3:7]  # PyBullet uses (x, y, z, w)
        vel = state[10:13]
        ang_vel = state[13:16]

        # Convert to (w, x, y, z) format
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Gate info
        gate = self.track.gates[self.current_gate_idx]
        gate_pos = gate.position
        gate_dir = self._get_gate_direction(gate)

        # Velocity target
        to_gate = gate_pos - pos
        to_gate_norm = to_gate / (np.linalg.norm(to_gate) + 1e-6)
        vel_target = to_gate_norm * self.target_velocity

        return {
            "position": pos,
            "velocity": vel,
            "orientation": quat_wxyz,
            "angular_velocity": ang_vel,
            "gate_position": gate_pos,
            "gate_direction": gate_dir,
            "velocity_target": vel_target,
        }

    def create_gate_visuals(self):
        """Create visual markers for gates in PyBullet."""
        for gate in self.track.gates:
            # Create gate frame visualization
            visual_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.05, gate.width/2, gate.height/2],
                rgbaColor=[1, 0.5, 0, 0.5],
            )

            body_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_id,
                basePosition=gate.position,
                baseOrientation=gate.orientation,
            )

            self.gate_visual_ids.append(body_id)


def create_monorace_track(num_gates: int = 11) -> TrackConfig:
    """
    Create a MonoRace-style circular track.

    Args:
        num_gates: Number of gates in the track

    Returns:
        TrackConfig for the track
    """
    gates = []
    radius = 3.0
    height_variation = 0.3
    base_height = 1.0

    for i in range(num_gates):
        angle = 2 * np.pi * i / num_gates
        next_angle = 2 * np.pi * (i + 1) / num_gates

        # Position on circle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = base_height + height_variation * np.sin(2 * angle)

        # Orientation: gate faces towards next gate
        direction = np.array([
            np.cos(next_angle) - np.cos(angle),
            np.sin(next_angle) - np.sin(angle),
            0,
        ])
        direction = direction / np.linalg.norm(direction)

        # Convert direction to quaternion (gate normal along x-axis)
        yaw = np.arctan2(direction[1], direction[0])
        quat = p.getQuaternionFromEuler([0, 0, yaw])

        gates.append(GateConfig(
            position=np.array([x, y, z]),
            orientation=np.array(quat),
        ))

    return TrackConfig(
        name=f"monorace_{num_gates}",
        gates=gates,
        start_position=np.array([radius, 0, 0.5]),
    )


if __name__ == "__main__":
    # Test environment
    print("Testing HighFreqRacingAviary...")

    # Create track
    track = create_monorace_track(5)
    print(f"Track: {track.name} with {len(track.gates)} gates")

    # Create environment
    env = HighFreqRacingAviary(
        track=track,
        ctrl_freq=500,
        pyb_freq=2000,
        gui=False,
    )

    print(f"Control frequency: {env.CTRL_FREQ} Hz")
    print(f"PyBullet frequency: {env.PYB_FREQ} Hz")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test episode
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")

    total_reward = 0
    for step in range(100):
        # Random action (hover-ish)
        action = np.ones(4) * 0.5 + np.random.uniform(-0.1, 0.1, 4)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Gates passed: {info['gates_passed']}")
    print(f"Final position: {info['position']}")

    env.close()

    # Test state for control
    env2 = HighFreqRacingAviary(track=track, gui=False)
    env2.reset()

    state = env2.get_state_for_control()
    print(f"\nState for control:")
    for key, value in state.items():
        print(f"  {key}: {value}")

    env2.close()

    print("\nTest complete!")
