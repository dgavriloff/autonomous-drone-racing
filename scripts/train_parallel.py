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


class VelocityRacingEnv(BaseRLAviary):
    """Racing environment with VELOCITY action space."""

    def __init__(
        self,
        track: TrackConfig,
        ctrl_freq: int = 48,
        pyb_freq: int = 240,
        gui: bool = False,
        gate_tolerance: float = 0.5,  # Increased from 0.3 - more forgiving during learning
        max_steps: int = 500,
        reward_gate: float = 50.0,
        reward_progress: float = 2.0,
        reward_velocity: float = 0.2,
        reward_smoothness: float = -0.01,
        reward_crash: float = -50.0,
    ):
        self.track = track
        self.gate_tolerance = gate_tolerance
        self.max_steps_custom = max_steps
        self.reward_gate = reward_gate
        self.reward_progress = reward_progress
        self.reward_velocity = reward_velocity
        self.reward_smoothness = reward_smoothness
        self.reward_crash = reward_crash
        self.current_gate = 0
        self.gates_passed = 0
        self.step_count = 0
        self.prev_dist = None
        self.prev_action = None

        super().__init__(
            drone_model=DroneModel.CF2X,
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

        # Altitude penalty - encourage staying at gate height
        altitude_error = abs(pos[2] - gate.position[2])
        reward -= 0.1 * altitude_error  # Penalize altitude drift

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


def make_env(rank, seed, num_gates, radius):
    """Create a single environment instance."""
    def _init():
        track = create_simple_track(num_gates, radius)
        env = VelocityRacingEnv(track, gui=False)
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
):
    """Train with parallel environments."""
    print("=" * 60)
    print("PARALLEL VELOCITY CONTROL TRAINING")
    print("=" * 60)
    print(f"Algorithm: {algorithm}")
    print(f"Parallel envs: {n_envs}")
    print(f"Gates: {num_gates}, Radius: {radius}m")
    print(f"Timesteps: {timesteps}")
    print()

    # Create parallel environments
    print(f"Creating {n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(i, 42, num_gates, radius) for i in range(n_envs)])
    env = VecMonitor(env)
    # Note: VecNormalize removed - relative observation space should be more stable

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Create model with larger batch for parallel training
    if algorithm.upper() == "SAC":
        # CRITICAL FIX: Use fixed entropy coefficient to prevent collapse
        # Research shows ent_coef="auto" can drop to 0.002 and kill exploration
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1000000,  # Larger buffer for parallel
            learning_starts=10000,  # More warmup for diverse initial data
            batch_size=512,  # Larger batch
            tau=0.005,
            gamma=0.99,
            ent_coef=0.1,  # FIXED: Prevent entropy collapse (was "auto")
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
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        num_gates=args.gates,
        radius=args.radius,
        n_envs=args.envs,
        algorithm=args.algorithm,
    )


if __name__ == "__main__":
    main()
