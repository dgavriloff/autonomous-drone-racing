#!/usr/bin/env python3
"""
Parallel training with multiple environments.

Uses SubprocVecEnv to run N environments in parallel, fully utilizing
multi-core CPUs. With 24 cores, we run 16 parallel envs.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

from src.envs.high_freq_racing import TrackConfig, GateConfig
import gymnasium as gym


class DomainRandomizationWrapper(gym.Wrapper):
    """Wrapper for domain randomization to improve sim-to-real transfer.

    Randomizes dynamics parameters at each reset:
    - Mass: affects hover thrust, acceleration
    - Thrust coefficient (KF): motor efficiency
    - Observation noise: position, velocity, orientation
    - Observation delay: simulates vision/sensor latency
    """

    def __init__(
        self,
        env,
        mass_range=(0.85, 1.15),  # ±15% of nominal
        kf_range=(0.88, 1.12),    # ±12% of nominal
        pos_noise=0.05,           # 5cm RMS position noise
        vel_noise=0.1,            # 0.1 m/s velocity noise
        ori_noise=0.05,           # ~3° orientation noise
        delay_frames=0,           # Observation delay in frames
        seed=None,
    ):
        super().__init__(env)
        self.mass_range = mass_range
        self.kf_range = kf_range
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.ori_noise = ori_noise
        self.delay_frames = delay_frames
        self.rng = np.random.default_rng(seed)

        # Store nominal values
        self.nominal_mass = env.MASS
        self.nominal_kf = env.KF

        # Observation buffer for delay
        self.obs_buffer = []

    def reset(self, **kwargs):
        # Randomize dynamics before reset
        self._randomize_dynamics()
        obs, info = self.env.reset(**kwargs)

        # Clear observation buffer
        self.obs_buffer = [obs] * (self.delay_frames + 1)

        # Add noise to observation
        obs = self._add_observation_noise(obs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update observation buffer
        self.obs_buffer.append(obs)
        if len(self.obs_buffer) > self.delay_frames + 1:
            self.obs_buffer.pop(0)

        # Return delayed and noisy observation
        delayed_obs = self.obs_buffer[0]
        noisy_obs = self._add_observation_noise(delayed_obs)

        return noisy_obs, reward, terminated, truncated, info

    def _randomize_dynamics(self):
        """Randomize drone physical parameters."""
        # Randomize mass
        mass_scale = self.rng.uniform(*self.mass_range)
        self.env.MASS = self.nominal_mass * mass_scale
        self.env.GRAVITY = self.env.MASS * self.env.G

        # Randomize thrust coefficient
        kf_scale = self.rng.uniform(*self.kf_range)
        self.env.KF = self.nominal_kf * kf_scale

        # Update hover RPM based on new parameters
        # hover_rpm = sqrt(mass * g / (4 * kf))
        self.env.HOVER_RPM = np.sqrt(self.env.GRAVITY / (4 * self.env.KF))

    def _add_observation_noise(self, obs):
        """Add Gaussian noise to observation."""
        noisy_obs = obs.copy()

        # Position noise (indices 0:3)
        if self.pos_noise > 0:
            noisy_obs[0:3] += self.rng.normal(0, self.pos_noise, 3)

        # Velocity noise (indices 3:6)
        if self.vel_noise > 0:
            noisy_obs[3:6] += self.rng.normal(0, self.vel_noise, 3)

        # Orientation noise (indices 6:9 for euler angles)
        if self.ori_noise > 0:
            noisy_obs[6:9] += self.rng.normal(0, self.ori_noise, 3)

        return noisy_obs


def create_simple_track(num_gates=5, radius=2.0, height=0.5):
    """Create a circular track for training."""
    gates = []
    for i in range(num_gates):
        angle = 2 * np.pi * i / num_gates
        next_angle = 2 * np.pi * (i + 1) / num_gates
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
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
    start_x = radius * 0.95
    return TrackConfig(
        name=f"circle_{num_gates}g_{radius}m",
        gates=gates,
        start_position=np.array([start_x, 0.0, height]),
    )


def create_competition_track(track_type="swift", scale=1.0, height_variation=0.0, seed=None):
    """Create competition-style tracks based on real racing layouts.

    Args:
        track_type: "swift" (7 gates, 75m lap), "figure8" (lemniscate), "random"
        scale: Scale factor for track dimensions (1.0 = full size)
        height_variation: Random height variation in meters (0 = flat)
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    gates = []

    if track_type == "swift":
        # Swift-style: 30x30m arena, 7 gates, 75m lap
        # Based on Nature 2023 paper track layout
        positions = [
            [12.0, 0.0],    # Gate 1: Start straight
            [8.0, 8.0],     # Gate 2: First turn
            [-2.0, 12.0],   # Gate 3: Far corner
            [-10.0, 5.0],   # Gate 4: Back straight
            [-8.0, -6.0],   # Gate 5: Second turn
            [0.0, -10.0],   # Gate 6: Far end
            [8.0, -4.0],    # Gate 7: Return to start
        ]
        base_height = 0.6

    elif track_type == "figure8":
        # Lemniscate (figure-8) pattern - common in TII dataset
        # Parametric: x = a*cos(t)/(1+sin²(t)), y = a*sin(t)*cos(t)/(1+sin²(t))
        a = 12.0  # Scale factor
        num_gates = 7
        positions = []
        for i in range(num_gates):
            t = 2 * np.pi * i / num_gates
            denom = 1 + np.sin(t)**2
            x = a * np.cos(t) / denom
            y = a * np.sin(t) * np.cos(t) / denom
            positions.append([x, y])
        base_height = 0.6

    elif track_type == "tii":
        # TII Racing style: 11 gates, 170m course (scaled down for pybullet)
        # Original is ~85m per lap, we scale to fit ~20m radius
        # Gate positions approximate the TII layout
        positions = [
            [8.0, 0.0],      # Gate 1: Start
            [6.0, 5.0],      # Gate 2: First turn
            [2.0, 8.0],      # Gate 3
            [-3.0, 9.0],     # Gate 4: Far corner
            [-7.0, 6.0],     # Gate 5
            [-9.0, 1.0],     # Gate 6: Back straight
            [-8.0, -4.0],    # Gate 7
            [-4.0, -7.0],    # Gate 8
            [1.0, -8.0],     # Gate 9: Bottom
            [5.0, -5.0],     # Gate 10
            [7.0, -2.0],     # Gate 11: Return
        ]
        base_height = 0.6

    elif track_type == "split_s":
        # Track with altitude changes (inspired by Swift Split-S)
        positions = [
            [6.0, 0.0],
            [4.0, 4.0],
            [0.0, 6.0],
            [-4.0, 4.0],
            [-6.0, 0.0],
            [-4.0, -4.0],
            [0.0, -6.0],
        ]
        base_height = 0.6
        # Heights will vary - handled below

    elif track_type == "random":
        # Random track within 30x30m bounds
        num_gates = rng.integers(5, 10)
        positions = []
        for _ in range(num_gates):
            x = rng.uniform(-12, 12)
            y = rng.uniform(-12, 12)
            positions.append([x, y])
        # Sort by angle from center for reasonable ordering
        positions.sort(key=lambda p: np.arctan2(p[1], p[0]))
        base_height = 0.6

    else:
        raise ValueError(f"Unknown track type: {track_type}")

    # Apply scale
    positions = [[p[0] * scale, p[1] * scale] for p in positions]

    # Create gates
    for i, (x, y) in enumerate(positions):
        # Height with optional variation
        z = base_height + rng.uniform(-height_variation, height_variation) if height_variation > 0 else base_height

        # Orientation: face toward next gate
        next_i = (i + 1) % len(positions)
        next_x, next_y = positions[next_i]
        direction = np.array([next_x - x, next_y - y, 0])
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        yaw = np.arctan2(direction[1], direction[0])
        quat = p.getQuaternionFromEuler([0, 0, yaw])

        gates.append(GateConfig(
            position=np.array([x, y, z]),
            orientation=np.array(quat),
        ))

    # Start position: slightly before first gate
    start_x = positions[0][0] * 0.8
    start_y = positions[0][1] * 0.8

    return TrackConfig(
        name=f"{track_type}_{len(gates)}g_s{scale:.1f}",
        gates=gates,
        start_position=np.array([start_x, start_y, base_height]),
    )


class VelocityRacingEnv(BaseRLAviary):
    """Racing environment with VELOCITY action space."""

    def __init__(
        self,
        track: TrackConfig,
        ctrl_freq: int = 48,
        pyb_freq: int = 240,
        gui: bool = False,
        gate_tolerance: float = 0.8,  # Larger tolerance so agent can pass more gates and learn
        max_steps: int = 500,
        reward_gate: float = 50.0,
        reward_progress: float = 2.0,
        reward_velocity: float = 0.2,
        reward_smoothness: float = -0.01,
        reward_crash: float = -50.0,
        speed_factor: float = 0.03,  # Fraction of MAX_SPEED_KMH for SPEED_LIMIT
    ):
        self.track = track
        self.gate_tolerance = gate_tolerance
        self.max_steps_custom = max_steps
        self.reward_gate = reward_gate
        self.reward_progress = reward_progress
        self.reward_velocity = reward_velocity
        self.reward_smoothness = reward_smoothness
        self.reward_crash = reward_crash
        self.speed_factor = speed_factor
        self.current_gate = 0
        self.gates_passed = 0
        self.step_count = 0
        self.prev_dist = None
        self.prev_action = None

        super().__init__(
            drone_model=DroneModel.CF2X,  # Crazyflie - reliable PID controller
            num_drones=1,
            initial_xyzs=np.array([track.start_position]),
            initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
        )

        # Override SPEED_LIMIT based on speed_factor
        # CF2X MAX_SPEED_KMH = 30 (8.33 m/s max physical capability)
        # speed_factor 1.0 = 8.33 m/s
        self.SPEED_LIMIT = self.speed_factor * self.MAX_SPEED_KMH * (1000/3600)

    def _observationSpace(self):
        # Full observation space with position (helps with track layout understanding)
        # [pos(3), vel(3), euler(3), ang_vel(3), to_gate_dir(3), dist] = 16 dims
        return spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        vel = state[10:13]
        ang_vel = state[13:16]
        euler = np.array(p.getEulerFromQuaternion(quat))
        gate_pos = self.track.gates[self.current_gate].position
        to_gate = gate_pos - pos
        dist = np.linalg.norm(to_gate)
        to_gate_dir = to_gate / (dist + 1e-6)
        obs = np.concatenate([pos, vel, euler, ang_vel, to_gate_dir, [dist]])
        return obs.astype(np.float32)

    def _computeReward(self):
        reward = 0.0
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        gate = self.track.gates[self.current_gate]
        dist = np.linalg.norm(pos - gate.position)

        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            reward += self.reward_progress * progress

        to_gate = gate.position - pos
        to_gate_dir = to_gate / (np.linalg.norm(to_gate) + 1e-6)
        vel_alignment = np.dot(vel, to_gate_dir)
        reward += self.reward_velocity * max(0, vel_alignment)

        if self.prev_action is not None and hasattr(self, 'last_clipped_action'):
            action_diff = np.linalg.norm(self.last_clipped_action - self.prev_action)
            reward += self.reward_smoothness * action_diff

        self.prev_dist = dist

        if dist < self.gate_tolerance and not gate.passed:
            reward += self.reward_gate
            gate.passed = True
            self.gates_passed += 1
            self.current_gate = min(self.current_gate + 1, len(self.track.gates) - 1)
            self.prev_dist = None

        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        euler = p.getEulerFromQuaternion(state[3:7])
        if pos[2] < 0.05:
            return True
        if np.abs(euler[0]) > 1.2 or np.abs(euler[1]) > 1.2:
            return True
        if self.gates_passed >= len(self.track.gates):
            return True
        return False

    def _computeTruncated(self):
        return self.step_count >= self.max_steps_custom

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        return {
            "position": state[0:3].copy(),
            "velocity": state[10:13].copy(),
            "speed": np.linalg.norm(state[10:13]),
            "gates_passed": self.gates_passed,
            "current_gate": self.current_gate,
        }

    def reset(self, seed=None, options=None):
        self.current_gate = 0
        self.gates_passed = 0
        self.step_count = 0
        self.prev_dist = None
        self.prev_action = None
        for gate in self.track.gates:
            gate.passed = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.step_count += 1
        obs, _, terminated, truncated, info = super().step(action)
        self.prev_action = self.last_clipped_action.copy() if hasattr(self, 'last_clipped_action') else None
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        if terminated and self.gates_passed < len(self.track.gates):
            reward += self.reward_crash
        info.update(self._computeInfo())
        return obs, reward, terminated, truncated, info


def make_env(rank, seed, num_gates, radius, max_steps=1000, gate_tolerance=0.8):
    """Create a single environment instance."""
    def _init():
        track = create_simple_track(num_gates, radius)
        env = VelocityRacingEnv(track, gui=False, max_steps=max_steps, gate_tolerance=gate_tolerance)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class ParallelProgressCallback(BaseCallback):
    """Track progress across parallel envs."""

    def __init__(self, n_envs, verbose=1):
        super().__init__(verbose)
        self.n_envs = n_envs
        self.best_gates = 0
        self.episode_gates = []

    def _on_step(self):
        # Check all envs for done episodes
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]
                gates = info.get("gates_passed", 0)
                self.episode_gates.append(gates)

                if gates > self.best_gates:
                    self.best_gates = gates
                    print(f"\n*** NEW BEST: {gates} gates! ***")

        if self.num_timesteps % 10000 == 0 and self.episode_gates:
            avg = np.mean(self.episode_gates[-100:])
            max_recent = max(self.episode_gates[-100:]) if self.episode_gates else 0
            print(f"\n[{self.num_timesteps}] Avg gates: {avg:.2f}, Max recent: {max_recent}, Best: {self.best_gates}")

        return True


def train(
    timesteps=1000000,
    num_gates=5,
    radius=1.5,
    n_envs=16,
    algorithm="SAC",
    max_steps=1000,
    gate_tolerance=0.8,
    resume_from=None,
):
    """Train with parallel environments."""
    print("=" * 60)
    print("PARALLEL VELOCITY CONTROL TRAINING")
    print("=" * 60)
    print(f"Algorithm: {algorithm}")
    print(f"Parallel envs: {n_envs}")
    print(f"Gates: {num_gates}, Radius: {radius}m")
    print(f"Gate tolerance: {gate_tolerance}m")
    print(f"Max steps per episode: {max_steps}")
    print(f"Timesteps: {timesteps}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print()

    # Create parallel environments
    print(f"Creating {n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(i, 42, num_gates, radius, max_steps, gate_tolerance) for i in range(n_envs)])
    env = VecMonitor(env)
    # Note: VecNormalize removed - relative observation space should be more stable

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Create model with larger batch for parallel training
    if resume_from:
        # Transfer learning: load pretrained model
        print(f"Loading pretrained model from {resume_from}...")
        if algorithm.upper() == "SAC":
            model = SAC.load(resume_from, env=env, verbose=1, tensorboard_log="./logs/parallel_vel")
        else:
            model = PPO.load(resume_from, env=env, verbose=1, tensorboard_log="./logs/parallel_vel")
    elif algorithm.upper() == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1000000,  # Larger buffer for parallel
            learning_starts=10000,  # More warmup for diverse initial data
            batch_size=512,  # Larger batch
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",  # Auto-tuning works better for this task
            verbose=1,
            tensorboard_log="./logs/parallel_vel",
        )
    else:
        # PPO benefits more from parallel envs and has stable exploration
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,  # Larger batch
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,  # Better credit assignment for multi-gate
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            verbose=1,
            tensorboard_log="./logs/parallel_vel",
        )

    # Callbacks
    progress_cb = ParallelProgressCallback(n_envs)
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="models/parallel_vel",
        name_prefix="parallel",
    )

    # Train
    print("Starting parallel training...")
    start = time.time()

    model.learn(
        total_timesteps=timesteps,
        callback=[progress_cb, checkpoint_cb],
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best gates: {progress_cb.best_gates}")
    print(f"Effective FPS: {timesteps / elapsed:.0f}")

    # Save
    Path("models/parallel_vel").mkdir(parents=True, exist_ok=True)
    model.save("models/parallel_vel/final")
    print("Model saved to models/parallel_vel/final")

    env.close()
    return model, progress_cb


def main():
    # Use spawn for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--gates", type=int, default=5)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--algorithm", type=str, default="SAC", choices=["SAC", "PPO"])
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode (default: 1000)")
    parser.add_argument("--tolerance", type=float, default=0.8, help="Gate tolerance in meters (default: 0.8)")
    parser.add_argument("--resume", type=str, default=None, help="Path to pretrained model for transfer learning")
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        num_gates=args.gates,
        radius=args.radius,
        n_envs=args.envs,
        algorithm=args.algorithm,
        max_steps=args.max_steps,
        gate_tolerance=args.tolerance,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
