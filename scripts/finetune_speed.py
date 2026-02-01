#!/usr/bin/env python3
"""
Fine-tune curriculum model for speed.

Key insight: Training from scratch doesn't work. The curriculum approach
is necessary to learn gate navigation.

This script fine-tunes the working curriculum_final model to go FASTER
while preserving gate completion ability.

Strategy:
1. Use original VelocityRacingEnv (not SpeedRacingEnv)
2. Add speed bonus to reward
3. Use VERY low learning rate (1e-5) to avoid catastrophic forgetting
4. Monitor both gates AND speed
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

from scripts.train_parallel import VelocityRacingEnv, create_simple_track


class SpeedVelocityRacingEnv(VelocityRacingEnv):
    """VelocityRacingEnv with added speed incentives."""

    def __init__(
        self,
        track,
        speed_bonus: float = 0.5,  # Bonus for velocity toward gate
        lap_time_bonus: float = 200.0,  # Bonus for fast completion
        **kwargs
    ):
        super().__init__(track, **kwargs)
        self.speed_bonus = speed_bonus
        self.lap_time_bonus = lap_time_bonus
        self.episode_speeds = []

    def _computeReward(self):
        # Original reward
        reward = super()._computeReward()

        # Add speed bonus: reward velocity TOWARD gate (not just raw speed)
        state = self._getDroneStateVector(0)
        vel = state[10:13]
        speed = np.linalg.norm(vel)
        self.episode_speeds.append(speed)

        gate = self.track.gates[self.current_gate]
        pos = state[0:3]
        to_gate = gate.position - pos
        to_gate_dir = to_gate / (np.linalg.norm(to_gate) + 1e-6)
        vel_toward_gate = np.dot(vel, to_gate_dir)

        # Extra speed bonus (on top of existing velocity reward)
        reward += self.speed_bonus * max(0, vel_toward_gate)

        return reward

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Add lap time bonus for fast completion
        if self.gates_passed >= len(self.track.gates):
            time_bonus = max(0, 1 - self.step_count / 1000) * self.lap_time_bonus
            reward += time_bonus

        # Add speed info
        info['avg_speed'] = np.mean(self.episode_speeds) if self.episode_speeds else 0
        info['speed'] = np.linalg.norm(self._getDroneStateVector(0)[10:13])

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.episode_speeds = []
        return super().reset(seed=seed, options=options)


def make_env(rank, seed, num_gates, radius, max_steps, gate_tolerance, speed_bonus, lap_time_bonus):
    """Create environment with speed incentives."""
    def _init():
        track = create_simple_track(num_gates, radius)
        env = SpeedVelocityRacingEnv(
            track,
            gui=False,
            max_steps=max_steps,
            gate_tolerance=gate_tolerance,
            speed_bonus=speed_bonus,
            lap_time_bonus=lap_time_bonus,
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class SpeedProgressCallback(BaseCallback):
    """Track both gates AND speed."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.best_gates = 0
        self.best_speed_at_5gates = 0
        self.episode_data = []

    def _on_step(self):
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]
                gates = info.get("gates_passed", 0)
                avg_speed = info.get("avg_speed", 0)

                self.episode_data.append({
                    'gates': gates,
                    'speed': avg_speed,
                })

                if gates > self.best_gates:
                    self.best_gates = gates
                    print(f"\n*** NEW BEST GATES: {gates}! ***")

                if gates >= 5 and avg_speed > self.best_speed_at_5gates:
                    self.best_speed_at_5gates = avg_speed
                    print(f"\n*** NEW BEST SPEED (5/5 gates): {avg_speed:.2f} m/s! ***")

        if self.num_timesteps % 10000 == 0 and self.episode_data:
            recent = self.episode_data[-100:]
            avg_gates = np.mean([d['gates'] for d in recent])
            avg_speed = np.mean([d['speed'] for d in recent])
            full_laps = sum(1 for d in recent if d['gates'] >= 5)
            print(f"\n[{self.num_timesteps}] Gates: {avg_gates:.2f}/5, "
                  f"Speed: {avg_speed:.2f} m/s, Full laps: {full_laps}/100")

        return True


def finetune(
    model_path="models/curriculum_final.zip",
    timesteps=500000,
    n_envs=16,
    max_steps=1000,
    gate_tolerance=0.8,
    speed_bonus=0.5,
    lap_time_bonus=200.0,
    learning_rate=1e-5,  # Very low LR for fine-tuning
):
    """Fine-tune curriculum model for speed."""
    print("=" * 60)
    print("SPEED FINE-TUNING")
    print("=" * 60)
    print(f"Base model: {model_path}")
    print(f"Learning rate: {learning_rate} (very low for fine-tuning)")
    print(f"Speed bonus: {speed_bonus}")
    print(f"Lap time bonus: {lap_time_bonus}")
    print(f"Gate tolerance: {gate_tolerance}m")
    print(f"Timesteps: {timesteps}")
    print()

    # Create environments
    env = SubprocVecEnv([
        make_env(i, 42, 5, 1.5, max_steps, gate_tolerance, speed_bonus, lap_time_bonus)
        for i in range(n_envs)
    ])
    env = VecMonitor(env)

    # Load pretrained model with new learning rate
    print(f"Loading {model_path}...")
    model = SAC.load(
        model_path,
        env=env,
        learning_rate=learning_rate,
        verbose=1,
        tensorboard_log="./logs/finetune_speed",
    )

    # Callbacks
    progress_cb = SpeedProgressCallback()
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="models/finetune_speed",
        name_prefix="speed",
    )

    # Fine-tune
    print("Starting fine-tuning...")
    start = time.time()

    model.learn(
        total_timesteps=timesteps,
        callback=[progress_cb, checkpoint_cb],
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\nFine-tuning completed in {elapsed/60:.1f} minutes")
    print(f"Best gates: {progress_cb.best_gates}")
    print(f"Best speed (at 5/5 gates): {progress_cb.best_speed_at_5gates:.2f} m/s")

    # Save
    Path("models/finetune_speed").mkdir(parents=True, exist_ok=True)
    model.save("models/finetune_speed/final")
    print("Model saved to models/finetune_speed/final")

    env.close()
    return model


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/curriculum_final.zip")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=0.8)
    parser.add_argument("--speed-bonus", type=float, default=0.5)
    parser.add_argument("--lap-bonus", type=float, default=200.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    finetune(
        model_path=args.model,
        timesteps=args.timesteps,
        n_envs=args.envs,
        max_steps=args.max_steps,
        gate_tolerance=args.tolerance,
        speed_bonus=args.speed_bonus,
        lap_time_bonus=args.lap_bonus,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
