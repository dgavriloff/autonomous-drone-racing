#!/usr/bin/env python3
"""
Swift-aligned PPO training.

Hyperparameters from Swift paper (Extended Data Table 1):
- Policy: 2-layer MLP, 128 nodes, LeakyReLU(0.2)
- Learning rate: 3e-4
- Training: 10^8 environment interactions
"""

import sys
from pathlib import Path
import numpy as np
import time
import argparse
import torch
import torch.nn as nn
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.envs.swift_racing_env import SwiftRacingEnv, create_simple_track


class SwiftFeaturesExtractor(BaseFeaturesExtractor):
    """
    Swift-style feature extractor.

    Simple pass-through for MLP policy - features are already meaningful.
    """
    def __init__(self, observation_space, features_dim=31):
        super().__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()

    def forward(self, observations):
        return self.flatten(observations)


class SwiftProgressCallback(BaseCallback):
    """Track training progress."""

    def __init__(self, n_envs, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.n_envs = n_envs
        self.eval_freq = eval_freq
        self.best_gates = 0
        self.episode_gates = []
        self.episode_rewards = []

    def _on_step(self):
        # Check for completed episodes
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]
                gates = info.get("gates_passed", 0)
                self.episode_gates.append(gates)

                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])

                if gates > self.best_gates:
                    self.best_gates = gates
                    if self.verbose:
                        print(f"\n*** NEW BEST: {gates} gates! ***")

        # Periodic logging
        if self.num_timesteps % self.eval_freq == 0 and self.episode_gates:
            recent_gates = self.episode_gates[-100:]
            avg = np.mean(recent_gates)
            max_recent = max(recent_gates)

            recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
            avg_reward = np.mean(recent_rewards)

            if self.verbose:
                print(f"\n[{self.num_timesteps:,}] "
                      f"Avg gates: {avg:.2f}, Max: {max_recent}, Best: {self.best_gates}, "
                      f"Avg reward: {avg_reward:.1f}")

        return True


def make_env(rank, seed, num_gates, radius, max_steps):
    """Create a single environment."""
    def _init():
        track = create_simple_track(num_gates, radius)
        env = SwiftRacingEnv(
            track=track,
            gui=False,
            max_steps=max_steps,
            gate_tolerance=0.8,
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train(
    timesteps: int = 10_000_000,
    num_gates: int = 5,
    radius: float = 1.5,
    n_envs: int = 16,
    max_steps: int = 1000,
    save_path: str = "models/swift_ppo",
    resume_from: str = None,
):
    """Train PPO with Swift hyperparameters."""
    print("=" * 60)
    print("SWIFT-ALIGNED PPO TRAINING")
    print("=" * 60)
    print(f"Algorithm: PPO")
    print(f"Parallel envs: {n_envs}")
    print(f"Gates: {num_gates}, Radius: {radius}m")
    print(f"Timesteps: {timesteps:,}")
    print(f"Action space: thrust + body rates")
    print(f"Observation: 31-dim Swift state")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print()

    # Create parallel environments
    print(f"Creating {n_envs} parallel environments...")
    env = SubprocVecEnv([
        make_env(i, 42, num_gates, radius, max_steps)
        for i in range(n_envs)
    ])
    env = VecMonitor(env)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Swift policy architecture
    # "2-layer MLP, 128 nodes each, LeakyReLU (α=0.2)"
    policy_kwargs = dict(
        features_extractor_class=SwiftFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=31),
        net_arch=dict(
            pi=[128, 128],  # Policy network
            vf=[128, 128],  # Value network (can use privileged info)
        ),
        activation_fn=nn.LeakyReLU,
    )

    if resume_from:
        print(f"Loading model from {resume_from}...")
        model = PPO.load(
            resume_from,
            env=env,
            verbose=1,
            tensorboard_log=None,  # Disabled
        )
    else:
        # Swift hyperparameters
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,  # Steps per env before update
            batch_size=64 * n_envs,  # Larger batch for parallel
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Entropy for exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=None,  # Disabled
        )

    print("Model architecture:")
    print(model.policy)
    print()

    # Callbacks
    progress_cb = SwiftProgressCallback(n_envs)
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path=save_path,
        name_prefix="swift_ppo",
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
    print(f"Effective FPS: {timesteps / elapsed:.0f}")

    # Save final model
    Path(save_path).mkdir(parents=True, exist_ok=True)
    final_path = f"{save_path}/final"
    model.save(final_path)
    print(f"Model saved to {final_path}")

    env.close()
    return model, progress_cb


def evaluate(model_path: str, num_episodes: int = 10, render: bool = False):
    """Evaluate trained model."""
    print("=" * 60)
    print("EVALUATING SWIFT PPO MODEL")
    print("=" * 60)

    model = PPO.load(model_path)

    track = create_simple_track(num_gates=5, radius=1.5)
    env = SwiftRacingEnv(track=track, gui=render, max_steps=1000)

    results = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        gates = info.get("gates_passed", 0)
        results.append(gates)
        print(f"Episode {ep+1}/{num_episodes}: {gates}/5 gates, reward={total_reward:.1f}")

    env.close()

    print()
    print(f"Average gates: {np.mean(results):.2f}/5")
    print(f"Success rate: {np.mean([r == 5 for r in results])*100:.0f}%")


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Swift-aligned PPO training")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--gates", type=int, default=5)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval", type=str, default=None, help="Evaluate model path")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, render=args.render)
    else:
        train(
            timesteps=args.timesteps,
            num_gates=args.gates,
            radius=args.radius,
            n_envs=args.envs,
            max_steps=args.max_steps,
            resume_from=args.resume,
        )


if __name__ == "__main__":
    main()
