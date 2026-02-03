"""
Swift-aligned racing environment.

Action space: [collective_thrust, roll_rate, pitch_rate, yaw_rate]
- collective_thrust: mass-normalized (m/s²), typical [0, 20] for ~2g
- body_rates: rad/s, typical [-10, 10]

Observation space: 31 dimensions (Swift paper)
- position (3)
- velocity (3)
- rotation matrix flattened (9) - NOT quaternion
- gate corners relative (4 corners × 3 = 12)
- previous action (4)
"""

import numpy as np
import pybullet as p
from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Optional

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


@dataclass
class GateConfig:
    """Gate configuration."""
    position: np.ndarray  # (3,) center position
    orientation: np.ndarray  # (4,) quaternion
    size: float = 0.5  # gate half-width
    passed: bool = False


@dataclass
class TrackConfig:
    """Track configuration."""
    name: str
    gates: List[GateConfig]
    start_position: np.ndarray


class SwiftRacingEnv(BaseAviary):
    """
    Swift-aligned racing environment.

    Key differences from VelocityRacingEnv:
    - Action: thrust + body rates (not velocity)
    - Observation: 31-dim Swift state (not 16-dim)
    - Uses rotation matrix (not quaternion) for attitude
    """

    def __init__(
        self,
        track: TrackConfig,
        ctrl_freq: int = 50,  # Swift used 50Hz control
        pyb_freq: int = 200,  # 4x physics substeps
        gui: bool = False,
        gate_tolerance: float = 0.8,
        max_steps: int = 1000,
        # Reward coefficients (FIXED - dense progress reward)
        lambda_progress: float = 50.0,  # Was 2.0 - NOW 50x per meter toward gate
        lambda_velocity: float = 2.0,   # NEW: reward velocity toward gate
        lambda_perception: float = 0.1,
        lambda_cmd_rate: float = -0.005,  # Reduced from -0.01
        lambda_cmd_smooth: float = -0.002,  # Reduced from -0.005
        crash_penalty: float = 20.0,  # Was 5.0 - moderate penalty
        gate_bonus: float = 100.0,  # Was 10.0 - big bonus for gate
        # Thrust scaling
        max_thrust_accel: float = 20.0,  # m/s², ~2g
        max_body_rate: float = 10.0,  # rad/s
    ):
        self.track = track
        self.gate_tolerance = gate_tolerance
        self.max_steps = max_steps

        # Reward coefficients
        self.lambda_progress = lambda_progress
        self.lambda_velocity = lambda_velocity
        self.lambda_perception = lambda_perception
        self.lambda_cmd_rate = lambda_cmd_rate
        self.lambda_cmd_smooth = lambda_cmd_smooth
        self.crash_penalty = crash_penalty
        self.gate_bonus = gate_bonus

        # Action scaling
        self.max_thrust_accel = max_thrust_accel
        self.max_body_rate = max_body_rate

        # State tracking
        self.current_gate = 0
        self.gates_passed = 0
        self.step_count = 0
        self.prev_dist = None
        self.prev_action = np.zeros(4)

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([track.start_position]),
            initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )

        # Gate corner offsets (local frame, square gate)
        self.gate_corner_offsets = np.array([
            [-0.5, -0.5, 0],  # bottom-left
            [0.5, -0.5, 0],   # bottom-right
            [0.5, 0.5, 0],    # top-right
            [-0.5, 0.5, 0],   # top-left
        ]) * gate_tolerance  # Scale by gate size

    def _actionSpace(self):
        """Swift action space: [thrust, roll_rate, pitch_rate, yaw_rate]."""
        # Normalized to [-1, 1], denormalized in _preprocessAction
        return spaces.Box(
            low=-np.ones((1, 4)),
            high=np.ones((1, 4)),
            dtype=np.float32
        )

    def _observationSpace(self):
        """Swift observation space: 31 dimensions."""
        # position(3) + velocity(3) + rotation_matrix(9) + gate_corners(12) + prev_action(4)
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, 31),
            dtype=np.float32
        )

    def _computeObs(self):
        """Compute Swift-style observation."""
        state = self._getDroneStateVector(0)

        # Position (3)
        pos = state[0:3]

        # Velocity (3)
        vel = state[10:13]

        # Rotation matrix from quaternion (9)
        quat = state[3:7]  # [x, y, z, w] in pybullet
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        rot_flat = rot_matrix.flatten()  # Row-major: R00, R01, R02, R10, ...

        # Gate corners relative to drone (12)
        gate = self.track.gates[self.current_gate]
        gate_corners_world = self._get_gate_corners_world(gate)
        gate_corners_rel = gate_corners_world - pos  # Relative to drone position
        # Transform to body frame
        gate_corners_body = (rot_matrix.T @ gate_corners_rel.T).T  # (4, 3)
        gate_corners_flat = gate_corners_body.flatten()  # (12,)

        # Previous action (4)
        prev_act = self.prev_action

        # Concatenate: 3 + 3 + 9 + 12 + 4 = 31
        obs = np.concatenate([
            pos,              # 3
            vel,              # 3
            rot_flat,         # 9
            gate_corners_flat, # 12
            prev_act,         # 4
        ]).astype(np.float32)

        return obs.reshape(1, 31)

    def _get_gate_corners_world(self, gate: GateConfig) -> np.ndarray:
        """Get gate corner positions in world frame."""
        # Rotate corner offsets by gate orientation
        rot_matrix = np.array(p.getMatrixFromQuaternion(gate.orientation)).reshape(3, 3)
        corners_world = gate.position + (rot_matrix @ self.gate_corner_offsets.T).T
        return corners_world  # (4, 3)

    def _preprocessAction(self, action):
        """Convert thrust + body rates to motor RPMs."""
        action = action.reshape(4)

        # Denormalize action
        # thrust: [-1, 1] -> [0, max_thrust_accel] (only positive thrust makes sense)
        thrust_accel = (action[0] + 1) / 2 * self.max_thrust_accel  # [0, max]

        # Body rates: [-1, 1] -> [-max_rate, max_rate]
        roll_rate = action[1] * self.max_body_rate
        pitch_rate = action[2] * self.max_body_rate
        yaw_rate = action[3] * self.max_body_rate

        # Convert thrust acceleration to total thrust force
        thrust_force = thrust_accel * self.M  # F = ma

        # Convert to motor RPMs using quadrotor mixer
        # For X configuration (CF2X):
        # Motor layout (looking from top):
        #   0(CW)   1(CCW)
        #      \ /
        #       X
        #      / \
        #   3(CCW)  2(CW)
        #
        # Thrust: all motors contribute equally
        # Roll: differential 0,3 vs 1,2
        # Pitch: differential 0,1 vs 2,3
        # Yaw: differential CW vs CCW (0,2 vs 1,3)

        # Base RPM for thrust (F = 4 * kf * rpm²)
        base_rpm_sq = thrust_force / (4 * self.KF)
        base_rpm = np.sqrt(np.maximum(base_rpm_sq, 0))

        # Body rate contributions (simplified linear mixer)
        # Scale factors for rate control (tuned for CF2X)
        arm_length = self.L  # Distance from center to motor

        # Torque needed for angular acceleration
        # τ = I * α, but for rate control we use proportional gain
        k_rate = 0.02 * self.HOVER_RPM  # Reduced for stability

        roll_diff = k_rate * roll_rate
        pitch_diff = k_rate * pitch_rate
        yaw_diff = k_rate * yaw_rate * 0.5  # Yaw

        # Motor mixing (CF2X X-config)
        # Motor positions (from URDF inspection):
        #   3(FL)   0(FR)     +X forward
        #      \ /
        #       X
        #      / \
        #   2(RL)   1(RR)
        #
        # Negative pitch command = pitch forward (nose down) = front faster
        # Positive roll command = roll right = right faster
        rpm = np.array([
            base_rpm - roll_diff + pitch_diff + yaw_diff,  # Motor 0: Front-Right, CW
            base_rpm - roll_diff - pitch_diff - yaw_diff,  # Motor 1: Rear-Right, CCW
            base_rpm + roll_diff - pitch_diff + yaw_diff,  # Motor 2: Rear-Left, CW
            base_rpm + roll_diff + pitch_diff - yaw_diff,  # Motor 3: Front-Left, CCW
        ])

        # Clip to valid RPM range
        rpm = np.clip(rpm, 0, self.MAX_RPM)

        return rpm.reshape(1, 4)

    def _computeReward(self):
        """Dense reward function - prioritizes progress toward gate."""
        reward = 0.0

        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        quat = state[3:7]

        gate = self.track.gates[self.current_gate]
        to_gate = gate.position - pos
        dist = np.linalg.norm(to_gate)
        to_gate_dir = to_gate / (dist + 1e-6)

        # 1. DENSE PROGRESS REWARD (most important!)
        # Reward getting closer to gate, penalize getting farther
        if self.prev_dist is not None:
            progress = self.prev_dist - dist  # Positive when approaching
            reward += self.lambda_progress * progress  # 50x per meter

        # 2. VELOCITY TOWARD GATE (reward speed in right direction)
        velocity_toward_gate = np.dot(vel, to_gate_dir)
        reward += self.lambda_velocity * max(0, velocity_toward_gate)  # Only reward positive velocity

        # 3. Perception reward: keep gate in camera FOV (minor)
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        camera_axis = rot_matrix[:, 0]  # Body x-axis in world frame
        cos_angle = np.dot(camera_axis, to_gate_dir)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        perception_reward = np.exp(-2.0 * angle**2)
        reward += self.lambda_perception * perception_reward

        # 4. Command smoothness (minor penalties)
        body_rates = state[13:16]
        reward += self.lambda_cmd_rate * np.linalg.norm(body_rates)

        if hasattr(self, 'last_action'):
            action_diff = np.linalg.norm(self.prev_action - self.last_action)
            reward += self.lambda_cmd_smooth * action_diff

        self.prev_dist = dist

        # 5. GATE PASSING BONUS (big reward!)
        if dist < self.gate_tolerance and not gate.passed:
            reward += self.gate_bonus  # +100
            gate.passed = True
            self.gates_passed += 1
            self.current_gate = min(self.current_gate + 1, len(self.track.gates) - 1)
            self.prev_dist = None  # Reset for next gate

        return reward

    def _computeTerminated(self):
        """Check if episode should terminate."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        euler = p.getEulerFromQuaternion(state[3:7])

        # Crashed (too low)
        if pos[2] < 0.05:
            return True

        # Flipped (extreme angles)
        if np.abs(euler[0]) > 1.5 or np.abs(euler[1]) > 1.5:  # ~86 degrees
            return True

        # Completed track
        if self.gates_passed >= len(self.track.gates):
            return True

        # Out of bounds
        if np.linalg.norm(pos[:2]) > 20.0 or pos[2] > 10.0:
            return True

        return False

    def _computeTruncated(self):
        """Check if episode should be truncated (time limit)."""
        return self.step_count >= self.max_steps

    def _computeInfo(self):
        """Return episode info."""
        state = self._getDroneStateVector(0)
        return {
            "position": state[0:3].copy(),
            "velocity": state[10:13].copy(),
            "gates_passed": self.gates_passed,
            "current_gate": self.current_gate,
            "step": self.step_count,
        }

    def reset(self, seed=None, options=None):
        """Reset environment."""
        self.current_gate = 0
        self.gates_passed = 0
        self.step_count = 0
        self.prev_dist = None
        self.prev_action = np.zeros(4)

        for gate in self.track.gates:
            gate.passed = False

        obs, info = super().reset(seed=seed, options=options)
        return self._computeObs(), self._computeInfo()

    def step(self, action):
        """Execute one step."""
        self.step_count += 1

        # Store action before processing
        self.last_action = self.prev_action.copy()
        self.prev_action = action.flatten()[:4].copy()

        # Use parent's step which handles physics
        obs, _, terminated, truncated, _ = super().step(action)

        # Compute our custom outputs
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        # Crash penalty
        if terminated and self.gates_passed < len(self.track.gates):
            reward -= self.crash_penalty

        return obs, reward, terminated, truncated, info


def create_simple_track(num_gates=5, radius=1.5, height=0.5):
    """Create a simple circular track."""
    gates = []
    for i in range(num_gates):
        angle = 2 * np.pi * i / num_gates
        next_angle = 2 * np.pi * (i + 1) / num_gates

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height

        # Gate faces toward next gate
        direction = np.array([
            np.cos(next_angle) - np.cos(angle),
            np.sin(next_angle) - np.sin(angle),
            0,
        ])
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        yaw = np.arctan2(direction[1], direction[0])
        quat = p.getQuaternionFromEuler([0, 0, yaw])

        gates.append(GateConfig(
            position=np.array([x, y, z]),
            orientation=np.array(quat),
        ))

    # Start position: center of circle, must fly to first gate
    # This ensures drone starts ~radius meters from first gate
    return TrackConfig(
        name=f"circle_{num_gates}g",
        gates=gates,
        start_position=np.array([0.0, 0.0, height]),
    )
