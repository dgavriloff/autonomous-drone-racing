#!/usr/bin/env python3
"""
Train with position-delta actions instead of raw motor RPMs.

The problem: Raw motor RPM actions require learning complex coordination
to produce directional movement. Hover (0.5, 0.5, 0.5, 0.5) doesn't move.

Solution: Use position deltas as actions, convert to motor RPMs via PID.
This makes the action space more intuitive - action [1,0,0] = move forward.

This is similar to how the PID baseline works and achieves 4.97 m/s.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import pybullet as p
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

from src.envs.high_freq_racing import TrackConfig, GateConfig


def create_simple_track(num_gates=3, spacing=1.0):
    """Create a simple straight-line track."""
    gates = []
    for i in range(num_gates):
        gates.append(GateConfig(
            position=np.array([spacing * (i + 1), 0.0, 0.5]),
            orientation=np.array([0, 0, 0, 1]),  # Facing forward
        ))

    return TrackConfig(
        name=f"straight_{num_gates}",
        gates=gates,
        start_position=np.array([0.0, 0.0, 0.5]),
    )


class PositionControlEnv(gym.Env):
    """
    Environment with position-delta actions.

    Action: [dx, dy, dz] - desired position delta (scaled)
    The PID controller converts this to motor RPMs.
    """

    def __init__(self, track, ctrl_freq=48, max_steps=500, gate_tolerance=0.3):
        super().__init__()

        self.track = track
        self.ctrl_freq = ctrl_freq
        self.max_steps = max_steps
        self.gate_tolerance = gate_tolerance

        # Action: position delta [-1, 1]^3, scaled to [-0.5, 0.5] meters
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation: [pos, vel, to_gate_dir, dist_to_gate] = 10D
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.env = None
        self.ctrl = None
        self.step_count = 0
        self.current_gate = 0
        self.gates_passed = 0

    def reset(self, seed=None, options=None):
        if self.env is not None:
            self.env.close()

        from gym_pybullet_drones.envs import CtrlAviary

        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([self.track.start_position]),
            initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=self.ctrl_freq,
            gui=False,
        )

        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.obs, _ = self.env.reset()

        self.step_count = 0
        self.current_gate = 0
        self.gates_passed = 0
        for gate in self.track.gates:
            gate.passed = False

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        state = self.obs[0]
        pos = state[0:3]
        vel = state[10:13]

        gate_pos = self.track.gates[self.current_gate].position
        to_gate = gate_pos - pos
        dist = np.linalg.norm(to_gate)
        to_gate_dir = to_gate / (dist + 1e-6)

        return np.concatenate([pos, vel, to_gate_dir, [dist]]).astype(np.float32)

    def _get_info(self):
        pos = self.obs[0, 0:3]
        vel = self.obs[0, 10:13]
        return {
            "position": pos,
            "velocity": vel,
            "speed": np.linalg.norm(vel),
            "gates_passed": self.gates_passed,
            "current_gate": self.current_gate,
        }

    def step(self, action):
        self.step_count += 1

        # Convert action to target position
        pos = self.obs[0, 0:3]
        delta = action * 0.5  # Scale [-1,1] to [-0.5, 0.5] meters
        target_pos = pos + delta

        # Get motor commands from PID
        rpm, _, _ = self.ctrl.computeControlFromState(
            control_timestep=1/self.ctrl_freq,
            state=self.obs[0],
            target_pos=target_pos,
            target_rpy=np.zeros(3),
            target_vel=np.zeros(3),
            target_rpy_rates=np.zeros(3),
        )

        # Step simulation
        self.obs, _, _, _, _ = self.env.step(np.array([rpm]))

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _compute_reward(self):
        reward = 0.0

        pos = self.obs[0, 0:3]
        vel = self.obs[0, 10:13]

        gate = self.track.gates[self.current_gate]
        dist = np.linalg.norm(pos - gate.position)

        # Progress reward: positive for moving towards gate
        to_gate = gate.position - pos
        to_gate_dir = to_gate / (np.linalg.norm(to_gate) + 1e-6)
        velocity_towards_gate = np.dot(vel, to_gate_dir)
        reward += 0.5 * velocity_towards_gate  # Reward forward velocity

        # Small penalty for distance
        reward -= 0.01 * dist

        # Gate passed bonus
        if dist < self.gate_tolerance and not gate.passed:
            reward += 100.0
            gate.passed = True
            self.gates_passed += 1
            self.current_gate = min(self.current_gate + 1, len(self.track.gates) - 1)

        return reward

    def _check_terminated(self):
        pos = self.obs[0, 0:3]

        # Crash
        if pos[2] < 0.05:
            return True

        # All gates passed
        if self.gates_passed >= len(self.track.gates):
            return True

        return False

    def close(self):
        if self.env is not None:
            self.env.close()


class ProgressCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.best_gates = 0
        self.episode_gates = []

    def _on_step(self):
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            gates = info.get("gates_passed", 0)
            self.episode_gates.append(gates)

            if gates > self.best_gates:
                self.best_gates = gates
                print(f"\n*** NEW BEST: {gates} gates! ***")

        if self.num_timesteps % 5000 == 0:
            if self.episode_gates:
                avg = np.mean(self.episode_gates[-50:])
                print(f"\n[{self.num_timesteps}] Avg gates: {avg:.2f}, Best: {self.best_gates}")

        return True


def train(timesteps=200000, num_gates=3, spacing=1.0):
    print("=" * 60)
    print("POSITION CONTROL TRAINING")
    print("=" * 60)
    print(f"Gates: {num_gates}, Spacing: {spacing}m")
    print()

    track = create_simple_track(num_gates, spacing)
    env = Monitor(PositionControlEnv(track))

    print(f"Track: {track.name}")
    for i, gate in enumerate(track.gates):
        print(f"  Gate {i}: {gate.position}")
    print()

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=500,
        batch_size=256,
        ent_coef="auto",
        verbose=1,
    )

    callback = ProgressCallback()

    print("Training...")
    start = time.time()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    print(f"\nTraining took {(time.time()-start)/60:.1f} minutes")
    print(f"Best gates: {callback.best_gates}")

    # Save
    Path("models/position_control").mkdir(parents=True, exist_ok=True)
    model.save("models/position_control/final")

    env.close()
    return model, callback


def test(model_path="models/position_control/final", num_episodes=5):
    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)

    track = create_simple_track(3, 1.0)
    env = PositionControlEnv(track)
    model = SAC.load(model_path)

    for ep in range(num_episodes):
        obs, info = env.reset()
        trajectory = []
        total_reward = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            trajectory.append(info["position"].copy())
            total_reward += reward

            if term or trunc:
                break

        traj = np.array(trajectory)
        travel = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        print(f"Ep {ep+1}: {info['gates_passed']}/{len(track.gates)} gates, "
              f"travel={travel:.2f}m, reward={total_reward:.1f}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--timesteps", type=int, default=200000)
    args = parser.parse_args()

    if args.train:
        train(timesteps=args.timesteps)
        test()
    elif args.test:
        test()
    else:
        train(timesteps=args.timesteps)
        test()


if __name__ == "__main__":
    main()
