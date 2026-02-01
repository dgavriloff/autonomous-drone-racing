#!/usr/bin/env python3
"""
Speed-optimized training for drone racing.

Current baseline: 0.25 m/s average speed
Target: 10+ m/s average speed (40x improvement)

Key changes from train_parallel.py:
1. Speed bonus reward (capped to prevent flying away)
2. Lap time bonus (faster completion = more reward)
3. Higher velocity alignment reward
4. Curriculum on minimum speed requirement
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

from src.envs.high_freq_racing import TrackConfig, GateConfig


def create_simple_track(num_gates=5, radius=1.5, height=0.5):
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


class SpeedRacingEnv(BaseRLAviary):
    """
    Racing environment optimized for SPEED.

    Key reward changes:
    - Strong speed bonus (up to target_speed)
    - Lap time bonus (faster = better)
    - Velocity alignment bonus
    - Gate completion still required
    """

    def __init__(
        self,
        track: TrackConfig,
        ctrl_freq: int = 48,
        pyb_freq: int = 240,
        gui: bool = False,
        gate_tolerance: float = 0.8,  # Same as original working env
        max_steps: int = 1000,
        # Speed-focused rewards (balanced to preserve gate completion)
        target_speed: float = 5.0,  # Target average speed
        reward_gate: float = 500.0,  # Very big bonus for gates (was 100)
        reward_speed: float = 0.1,  # Modest reward for speed (was 0.5)
        reward_progress: float = 2.0,  # Reward for approaching gate
        reward_alignment: float = 0.5,  # Reward for velocity toward gate (was 0.3)
        reward_lap_time: float = 500.0,  # Big bonus for fast lap (was 200)
        reward_crash: float = -200.0,  # Stronger crash penalty (was -100)
        # Speed training modes
        reward_mode: str = "default",  # default, min_speed, massive_lap, curriculum
        min_speed: float = 0.5,  # Min speed for penalty mode
        min_speed_penalty: float = -1.0,  # Penalty per step below min_speed
    ):
        self.track = track
        self.gate_tolerance = gate_tolerance
        self.max_steps_custom = max_steps
        self.target_speed = target_speed
        self.reward_gate = reward_gate
        self.reward_speed = reward_speed
        self.reward_progress = reward_progress
        self.reward_alignment = reward_alignment
        self.reward_lap_time = reward_lap_time
        self.reward_crash = reward_crash
        self.reward_mode = reward_mode
        self.min_speed = min_speed
        self.min_speed_penalty = min_speed_penalty

        self.current_gate = 0
        self.gates_passed = 0
        self.step_count = 0
        self.prev_dist = None
        self.prev_action = None
        self.episode_speeds = []

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
        speed = np.linalg.norm(vel)
        self.episode_speeds.append(speed)

        gate = self.track.gates[self.current_gate]
        dist = np.linalg.norm(pos - gate.position)

        # 1. Progress reward (getting closer to gate)
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            reward += self.reward_progress * progress

        # 2. Velocity alignment (moving TOWARD gate, not just fast)
        to_gate = gate.position - pos
        to_gate_dir = to_gate / (np.linalg.norm(to_gate) + 1e-6)
        vel_toward_gate = np.dot(vel, to_gate_dir)
        reward += self.reward_alignment * max(0, vel_toward_gate)

        # 3. Speed bonus (capped at target_speed to prevent flying away)
        # Only reward speed up to target, diminishing returns above
        speed_bonus = min(speed, self.target_speed) / self.target_speed
        reward += self.reward_speed * speed_bonus

        # 4. Mode-specific rewards
        if self.reward_mode == "min_speed":
            # Penalize going too slow
            if speed < self.min_speed:
                reward += self.min_speed_penalty
        elif self.reward_mode == "massive_lap":
            # Extra alignment bonus - really push toward gate
            reward += 1.0 * max(0, vel_toward_gate)

        self.prev_dist = dist

        # 4. Gate passage reward
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

        # Crash conditions
        if pos[2] < 0.05:
            return True
        if np.abs(euler[0]) > 1.2 or np.abs(euler[1]) > 1.2:
            return True

        # Success: completed all gates
        if self.gates_passed >= len(self.track.gates):
            return True

        return False

    def _computeTruncated(self):
        return self.step_count >= self.max_steps_custom

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        avg_speed = np.mean(self.episode_speeds) if self.episode_speeds else 0
        return {
            "position": state[0:3].copy(),
            "velocity": state[10:13].copy(),
            "speed": np.linalg.norm(state[10:13]),
            "avg_speed": avg_speed,
            "gates_passed": self.gates_passed,
            "current_gate": self.current_gate,
            "steps": self.step_count,
        }

    def reset(self, seed=None, options=None):
        self.current_gate = 0
        self.gates_passed = 0
        self.step_count = 0
        self.prev_dist = None
        self.prev_action = None
        self.episode_speeds = []
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

        # Crash penalty
        if terminated and self.gates_passed < len(self.track.gates):
            state = self._getDroneStateVector(0)
            pos = state[0:3]
            if pos[2] < 0.05:  # Actually crashed
                reward += self.reward_crash

        # Lap time bonus: reward for completing faster
        if self.gates_passed >= len(self.track.gates):
            # Bonus inversely proportional to steps taken
            # Max bonus at 200 steps, zero bonus at 1000 steps
            lap_bonus = self.reward_lap_time
            if self.reward_mode == "massive_lap":
                lap_bonus = 5000.0  # Massive bonus for fast completion
            time_bonus = max(0, 1 - self.step_count / 1000) * lap_bonus
            reward += time_bonus

        info.update(self._computeInfo())
        return obs, reward, terminated, truncated, info


def make_env(rank, seed, num_gates, radius, max_steps, gate_tolerance, target_speed, reward_mode="default"):
    """Create a single environment instance."""
    def _init():
        track = create_simple_track(num_gates, radius)
        env = SpeedRacingEnv(
            track,
            gui=False,
            max_steps=max_steps,
            gate_tolerance=gate_tolerance,
            target_speed=target_speed,
            reward_mode=reward_mode,
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class SpeedProgressCallback(BaseCallback):
    """Track speed and gates across parallel envs."""

    def __init__(self, n_envs, verbose=1):
        super().__init__(verbose)
        self.n_envs = n_envs
        self.best_gates = 0
        self.best_speed = 0
        self.episode_speeds = []
        self.episode_gates = []

    def _on_step(self):
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]
                gates = info.get("gates_passed", 0)
                avg_speed = info.get("avg_speed", 0)

                self.episode_gates.append(gates)
                self.episode_speeds.append(avg_speed)

                if gates > self.best_gates:
                    self.best_gates = gates
                    print(f"\n*** NEW BEST GATES: {gates}! ***")

                if avg_speed > self.best_speed and gates >= 5:
                    self.best_speed = avg_speed
                    print(f"\n*** NEW BEST SPEED (5/5): {avg_speed:.2f} m/s! ***")

        if self.num_timesteps % 10000 == 0 and self.episode_speeds:
            recent_speeds = self.episode_speeds[-100:]
            recent_gates = self.episode_gates[-100:]
            avg_speed = np.mean(recent_speeds)
            avg_gates = np.mean(recent_gates)
            full_laps = sum(1 for g in recent_gates if g >= 5)
            print(f"\n[{self.num_timesteps}] Avg speed: {avg_speed:.2f} m/s, "
                  f"Avg gates: {avg_gates:.2f}, Full laps: {full_laps}/100")

        return True


def train(
    timesteps=1000000,
    num_gates=5,
    radius=1.5,
    n_envs=16,
    max_steps=500,
    gate_tolerance=0.8,
    target_speed=5.0,
    resume_from=None,
    learning_rate=3e-4,
    reward_mode="default",
):
    """Train speed-optimized policy."""
    print("=" * 60)
    print("SPEED-OPTIMIZED TRAINING")
    print("=" * 60)
    print(f"Target speed: {target_speed} m/s")
    print(f"Parallel envs: {n_envs}")
    print(f"Gates: {num_gates}, Radius: {radius}m")
    print(f"Gate tolerance: {gate_tolerance}m")
    print(f"Max steps: {max_steps}")
    print(f"Timesteps: {timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Reward mode: {reward_mode}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
        print("TIP: Use --lr 3e-5 for fine-tuning to avoid catastrophic forgetting")
    print()

    # Create parallel environments
    env = SubprocVecEnv([
        make_env(i, 42, num_gates, radius, max_steps, gate_tolerance, target_speed, reward_mode)
        for i in range(n_envs)
    ])
    env = VecMonitor(env)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Create or load model
    if resume_from:
        print(f"Loading pretrained model from {resume_from}...")
        model = SAC.load(
            resume_from,
            env=env,
            verbose=1,
            tensorboard_log="./logs/speed",
            learning_rate=learning_rate,  # Override LR for fine-tuning
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            verbose=1,
            tensorboard_log="./logs/speed",
        )

    # Callbacks
    progress_cb = SpeedProgressCallback(n_envs)
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="models/speed",
        name_prefix="speed",
    )

    # Train
    print("Starting speed training...")
    start = time.time()

    model.learn(
        total_timesteps=timesteps,
        callback=[progress_cb, checkpoint_cb],
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best gates: {progress_cb.best_gates}")
    print(f"Best speed (with 5/5 gates): {progress_cb.best_speed:.2f} m/s")

    # Save
    Path("models/speed").mkdir(parents=True, exist_ok=True)
    model.save("models/speed/final")
    print("Model saved to models/speed/final")

    env.close()
    return model, progress_cb


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--gates", type=int, default=5)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=0.8)
    parser.add_argument("--target-speed", type=float, default=5.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (use 3e-5 for fine-tuning)")
    parser.add_argument("--mode", type=str, default="default",
                        choices=["default", "min_speed", "massive_lap"],
                        help="Reward mode: default, min_speed, massive_lap")
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        num_gates=args.gates,
        radius=args.radius,
        n_envs=args.envs,
        max_steps=args.max_steps,
        gate_tolerance=args.tolerance,
        target_speed=args.target_speed,
        resume_from=args.resume,
        learning_rate=args.lr,
        reward_mode=args.mode,
    )


if __name__ == "__main__":
    main()
