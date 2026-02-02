#!/usr/bin/env python3
"""Evaluate the teacher model (curriculum_final.zip)."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, SAC
from scripts.train_parallel import VelocityRacingEnv, create_simple_track


def evaluate_teacher(model_path: str, num_episodes: int = 10, num_gates: int = 5, render: bool = False):
    """Evaluate teacher model on gate-passing task."""
    print("=" * 60)
    print("EVALUATING TEACHER MODEL")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Gates: {num_gates}")
    print()

    # Load model (SAC - has actor/critic optimizers in zip)
    model = SAC.load(model_path)
    print(f"Model loaded successfully")
    print(f"Policy: {model.policy}")
    print()

    # Create environment
    track = create_simple_track(num_gates=num_gates, radius=1.5)
    env = VelocityRacingEnv(
        track=track,
        gui=render,
        max_steps=1000,
        gate_tolerance=0.8,
    )

    # Evaluate
    results = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        gates_passed = info.get("gates_passed", 0)
        results.append({
            "gates_passed": gates_passed,
            "reward": total_reward,
            "steps": steps,
        })
        print(f"Episode {ep+1}/{num_episodes}: {gates_passed}/{num_gates} gates, reward={total_reward:.1f}")

    env.close()

    # Summary
    avg_gates = np.mean([r["gates_passed"] for r in results])
    success_rate = np.mean([r["gates_passed"] == num_gates for r in results])

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average gates: {avg_gates:.2f}/{num_gates}")
    print(f"Success rate: {success_rate*100:.0f}%")
    print()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/curriculum_final.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--gates", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    evaluate_teacher(args.model, args.episodes, args.gates, args.render)
