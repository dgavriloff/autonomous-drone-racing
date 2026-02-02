#!/usr/bin/env python3
"""
Quick RL fine-tuning test.
Train for just 50K steps to see if RL can improve the BC model.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from scripts.train_parallel import VelocityRacingEnv, create_simple_track


class QuickEvalCallback(BaseCallback):
    """Evaluate every N steps."""
    def __init__(self, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.results = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Quick eval
            env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
            gates_passed = []
            for _ in range(5):
                obs, _ = self.model.env.reset()
                done = False
                ep_gates = 0
                steps = 0
                while not done and steps < 500:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.model.env.step(action)
                    done = terminated.any() if hasattr(terminated, 'any') else terminated
                    steps += 1
                # Get gates from info
                if isinstance(info, list):
                    ep_gates = info[0].get('gates_passed', 0)
                else:
                    ep_gates = info.get('gates_passed', 0)
                gates_passed.append(ep_gates)
            avg = np.mean(gates_passed)
            self.results.append((self.n_calls, avg))
            print(f"  Step {self.n_calls}: {avg:.1f}/5 gates")
        return True


def make_env():
    def _init():
        track = create_simple_track(num_gates=5, radius=1.5)
        return VelocityRacingEnv(
            track=track,
            ctrl_freq=48,
            pyb_freq=240,
            gui=False,
            gate_tolerance=0.8,
            max_steps=500,
        )
    return _init


def quick_rl_test(timesteps=50000, n_envs=8):
    print("=" * 60)
    print("QUICK RL TEST")
    print("=" * 60)
    print(f"Training for {timesteps} steps with {n_envs} parallel envs")
    print("This tests if RL can improve on BC baseline\n")

    # Create parallel envs
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    # Create PPO model (no BC initialization - just test if RL works at all)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        device="cuda",
    )

    # Quick baseline (random policy)
    print("Baseline (untrained): ~0 gates (random)")
    callback = QuickEvalCallback(eval_freq=10000)

    # Train
    print(f"\nTraining {timesteps} steps...")
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

    # Final eval
    print("\nFinal evaluation (10 episodes):")
    gates_list = []
    for ep in range(10):
        obs = env.reset()
        done = [False] * n_envs
        ep_gates = 0
        steps = 0
        while not all(done) and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            steps += 1
        ep_gates = info[0].get('gates_passed', 0)
        gates_list.append(ep_gates)
        print(f"  Ep {ep+1}: {ep_gates}/5 gates")

    env.close()

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"Average gates: {np.mean(gates_list):.1f}/5")

    if np.mean(gates_list) > 0.5:
        print("\n✓ RL IS LEARNING - fine-tuning BC model should help!")
    else:
        print("\n✗ RL struggling from scratch - may need more steps or tuning")

    return callback.results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--envs", type=int, default=8)
    args = parser.parse_args()

    quick_rl_test(args.timesteps, args.envs)
