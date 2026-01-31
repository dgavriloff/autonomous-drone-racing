#!/usr/bin/env python3
"""
Train SAC agent with VELOCITY action space.

The problem: Raw motor RPM actions (ActionType.RPM) require learning complex
motor coordination - hover (0.5, 0.5, 0.5, 0.5) doesn't produce movement.

Solution: Use ActionType.VEL which provides velocity commands. The built-in
PID controller handles motor coordination, so action [1,0,0,0] = move forward.

Based on research:
- gym-pybullet-drones ActionType.VEL abstracts away motor control
- Swift/Nature paper uses thrust + body rates (similar abstraction level)
- PPO works well with 10M+ timesteps, SAC for faster exploration
- Reward: progress + velocity alignment + smoothness + gate bonus

Usage:
    python scripts/train_velocity_control.py --train
    python scripts/train_velocity_control.py --test
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

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

        # Gate faces towards next gate
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

    # Start at first gate position
    start_x = radius * 0.95
    start_y = 0.0
    start_z = height

    return TrackConfig(
        name=f"circle_{num_gates}g_{radius}m",
        gates=gates,
        start_position=np.array([start_x, start_y, start_z]),
    )


class VelocityRacingEnv(BaseRLAviary):
    """
    Racing environment with VELOCITY action space.

    This uses ActionType.VEL which provides:
    - Action: [vx, vy, vz, yaw_rate] in normalized [-1, 1] range
    - Built-in PID converts to motor RPMs
    - Much easier to learn than raw motor control
    """

    def __init__(
        self,
        track: TrackConfig,
        ctrl_freq: int = 48,  # Standard control frequency for VEL mode
        pyb_freq: int = 240,  # Standard PyBullet frequency
        gui: bool = False,
        gate_tolerance: float = 0.3,
        max_steps: int = 500,
        # Reward weights (based on Swift paper approach)
        reward_gate: float = 50.0,
        reward_progress: float = 2.0,  # PBRS scale
        reward_velocity: float = 0.2,  # Velocity alignment bonus
        reward_smoothness: float = -0.01,  # Action smoothness penalty
        reward_crash: float = -50.0,
    ):
        self.track = track
        self.gate_tolerance = gate_tolerance
        self.max_steps_custom = max_steps

        # Reward weights
        self.reward_gate = reward_gate
        self.reward_progress = reward_progress
        self.reward_velocity = reward_velocity
        self.reward_smoothness = reward_smoothness
        self.reward_crash = reward_crash

        # State tracking
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
            act=ActionType.VEL,  # VELOCITY CONTROL - key change!
        )

    def _observationSpace(self):
        """
        Observation: [pos(3), vel(3), euler(3), ang_vel(3), to_gate(3), dist(1)] = 16D
        """
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

    def _computeObs(self):
        """Compute observation with gate-relative information."""
        state = self._getDroneStateVector(0)

        pos = state[0:3]
        quat = state[3:7]
        vel = state[10:13]
        ang_vel = state[13:16]

        # Euler angles
        euler = np.array(p.getEulerFromQuaternion(quat))

        # Gate-relative info
        gate_pos = self.track.gates[self.current_gate].position
        to_gate = gate_pos - pos
        dist = np.linalg.norm(to_gate)
        to_gate_dir = to_gate / (dist + 1e-6)

        obs = np.concatenate([
            pos,           # 3
            vel,           # 3
            euler,         # 3
            ang_vel,       # 3
            to_gate_dir,   # 3
            [dist],        # 1
        ])

        return obs.astype(np.float32)

    def _computeReward(self):
        """
        Compute reward based on Swift paper approach:
        - Progress towards gate (PBRS)
        - Velocity alignment bonus
        - Action smoothness penalty
        - Gate passage bonus
        """
        reward = 0.0

        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]

        gate = self.track.gates[self.current_gate]
        dist = np.linalg.norm(pos - gate.position)

        # 1. Progress reward (PBRS)
        if self.prev_dist is not None:
            progress = self.prev_dist - dist  # Positive if getting closer
            reward += self.reward_progress * progress

        # 2. Velocity alignment bonus
        to_gate = gate.position - pos
        to_gate_dir = to_gate / (np.linalg.norm(to_gate) + 1e-6)
        vel_alignment = np.dot(vel, to_gate_dir)
        reward += self.reward_velocity * max(0, vel_alignment)

        # 3. Action smoothness penalty
        if self.prev_action is not None:
            action_diff = np.linalg.norm(self.last_clipped_action - self.prev_action)
            reward += self.reward_smoothness * action_diff

        # Update tracking
        self.prev_dist = dist

        # 4. Gate passage bonus
        if dist < self.gate_tolerance and not gate.passed:
            reward += self.reward_gate
            gate.passed = True
            self.gates_passed += 1
            self.current_gate = min(self.current_gate + 1, len(self.track.gates) - 1)
            self.prev_dist = None  # Reset for new gate

        return reward

    def _computeTerminated(self):
        """Check for crash or completion."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        euler = p.getEulerFromQuaternion(state[3:7])

        # Crash: ground contact
        if pos[2] < 0.05:
            return True

        # Crash: excessive tilt
        if np.abs(euler[0]) > 1.2 or np.abs(euler[1]) > 1.2:
            return True

        # Success: all gates passed
        if self.gates_passed >= len(self.track.gates):
            return True

        return False

    def _computeTruncated(self):
        """Check for timeout."""
        return self.step_count >= self.max_steps_custom

    def _computeInfo(self):
        """Return info dict."""
        state = self._getDroneStateVector(0)
        return {
            "position": state[0:3].copy(),
            "velocity": state[10:13].copy(),
            "speed": np.linalg.norm(state[10:13]),
            "gates_passed": self.gates_passed,
            "current_gate": self.current_gate,
        }

    def reset(self, seed=None, options=None):
        """Reset environment."""
        self.current_gate = 0
        self.gates_passed = 0
        self.step_count = 0
        self.prev_dist = None
        self.prev_action = None

        for gate in self.track.gates:
            gate.passed = False

        return super().reset(seed=seed, options=options)

    def step(self, action):
        """Step with action tracking."""
        self.step_count += 1

        obs, _, terminated, truncated, info = super().step(action)

        # Store action for smoothness computation
        self.prev_action = self.last_clipped_action.copy() if hasattr(self, 'last_clipped_action') else None

        # Custom reward
        reward = self._computeReward()

        # Crash penalty
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()

        if terminated and self.gates_passed < len(self.track.gates):
            reward += self.reward_crash

        info.update(self._computeInfo())

        return obs, reward, terminated, truncated, info


class ProgressCallback(BaseCallback):
    """Track training progress."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.best_gates = 0
        self.episode_gates = []
        self.episode_rewards = []

    def _on_step(self):
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            gates = info.get("gates_passed", 0)
            self.episode_gates.append(gates)

            if gates > self.best_gates:
                self.best_gates = gates
                print(f"\n*** NEW BEST: {gates} gates! ***")

        if self.num_timesteps % 10000 == 0 and self.episode_gates:
            avg = np.mean(self.episode_gates[-100:])
            max_recent = max(self.episode_gates[-100:]) if self.episode_gates else 0
            print(f"\n[{self.num_timesteps}] Avg gates: {avg:.2f}, Recent max: {max_recent}, Best: {self.best_gates}")

        return True


def train(
    timesteps=500000,
    num_gates=5,
    radius=1.5,  # Start with smaller radius for easier learning
    algorithm="SAC",
):
    """Train with velocity control."""
    print("=" * 60)
    print("VELOCITY CONTROL TRAINING")
    print("=" * 60)
    print(f"Algorithm: {algorithm}")
    print(f"Gates: {num_gates}, Radius: {radius}m")
    print(f"Timesteps: {timesteps}")
    print()
    print("Key insight: Using ActionType.VEL instead of ActionType.RPM")
    print("The PID handles motor coordination, making learning much easier!")
    print()

    # Create track
    track = create_simple_track(num_gates, radius)

    print(f"Track: {track.name}")
    print(f"Start: {track.start_position}")
    for i, gate in enumerate(track.gates):
        print(f"  Gate {i}: {gate.position}")
    print()

    # Create environment
    env = Monitor(VelocityRacingEnv(track, gui=False))

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Create model
    if algorithm.upper() == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            verbose=1,
            tensorboard_log="./logs/velocity_control",
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./logs/velocity_control",
        )

    # Callbacks
    progress_cb = ProgressCallback()
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="models/velocity_control",
        name_prefix="vel_ctrl",
    )

    # Train
    print("Starting training...")
    start = time.time()

    model.learn(
        total_timesteps=timesteps,
        callback=[progress_cb, checkpoint_cb],
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best gates: {progress_cb.best_gates}")

    # Save
    Path("models/velocity_control").mkdir(parents=True, exist_ok=True)
    model.save("models/velocity_control/final")
    print("Model saved to models/velocity_control/final")

    env.close()

    return model, progress_cb


def test(model_path="models/velocity_control/final", num_episodes=10, num_gates=5, radius=1.5):
    """Test trained model."""
    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)

    track = create_simple_track(num_gates, radius)
    env = VelocityRacingEnv(track, gui=False)

    model = SAC.load(model_path)

    results = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        trajectory = []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            trajectory.append(info["position"].copy())

            if term or trunc:
                break

        traj = np.array(trajectory)
        h_dist = np.linalg.norm(traj[-1, :2] - traj[0, :2]) if len(traj) > 1 else 0
        v_dist = abs(traj[-1, 2] - traj[0, 2]) if len(traj) > 1 else 0

        results.append({
            "gates": info["gates_passed"],
            "reward": total_reward,
            "h_travel": h_dist,
            "v_travel": v_dist,
        })

        direction = "HORIZONTAL" if h_dist > v_dist else "VERTICAL"
        print(f"Ep {ep+1}: {info['gates_passed']}/{num_gates} gates, reward={total_reward:.1f}, "
              f"{direction} (h={h_dist:.2f}m, v={v_dist:.2f}m)")

    env.close()

    # Summary
    avg_gates = np.mean([r["gates"] for r in results])
    max_gates = max([r["gates"] for r in results])
    avg_h = np.mean([r["h_travel"] for r in results])
    avg_v = np.mean([r["v_travel"] for r in results])

    print(f"\nAverage gates: {avg_gates:.2f}/{num_gates}")
    print(f"Best gates: {max_gates}/{num_gates}")
    print(f"Avg horizontal travel: {avg_h:.2f}m")
    print(f"Avg vertical travel: {avg_v:.2f}m")

    if avg_h > avg_v:
        print("\n>>> Agent navigating HORIZONTALLY (good!)")
    else:
        print("\n>>> Agent still moving VERTICALLY (needs work)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--gates", type=int, default=5)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--algorithm", type=str, default="SAC", choices=["SAC", "PPO"])
    args = parser.parse_args()

    if args.train:
        train(
            timesteps=args.timesteps,
            num_gates=args.gates,
            radius=args.radius,
            algorithm=args.algorithm,
        )
        print("\nAuto-testing...")
        test(num_gates=args.gates, radius=args.radius)
    elif args.test:
        test(num_gates=args.gates, radius=args.radius)
    else:
        # Default: train and test
        train(
            timesteps=args.timesteps,
            num_gates=args.gates,
            radius=args.radius,
            algorithm=args.algorithm,
        )
        test(num_gates=args.gates, radius=args.radius)


if __name__ == "__main__":
    main()
