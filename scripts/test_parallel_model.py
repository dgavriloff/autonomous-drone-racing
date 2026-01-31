#!/usr/bin/env python3
"""Test the parallel-trained velocity control model."""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from scripts.train_parallel import VelocityRacingEnv, create_simple_track


def test_model(model_path="models/parallel_vel/final", num_episodes=5, gui=True):
    """Test the trained model."""
    print("=" * 60)
    print("TESTING PARALLEL-TRAINED MODEL")
    print("=" * 60)

    # Create environment
    track = create_simple_track(num_gates=5, radius=1.5)
    env = VelocityRacingEnv(track, gui=gui, max_steps=500)

    # Load model
    model = SAC.load(model_path)
    print(f"Loaded model from {model_path}")
    print()

    # Run episodes
    results = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if gui:
                time.sleep(1/48)  # Match control frequency for visualization

        gates = info.get("gates_passed", 0)
        speed = info.get("speed", 0)
        results.append({
            "episode": ep + 1,
            "gates": gates,
            "reward": total_reward,
            "steps": steps,
            "final_speed": speed,
        })
        print(f"Episode {ep+1}: {gates}/5 gates, reward={total_reward:.1f}, steps={steps}")

    env.close()

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_gates = np.mean([r["gates"] for r in results])
    max_gates = max(r["gates"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])
    print(f"Average gates: {avg_gates:.2f}/5")
    print(f"Max gates: {max_gates}/5")
    print(f"Average reward: {avg_reward:.1f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/parallel_vel/final")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    test_model(args.model, args.episodes, gui=not args.no_gui)
