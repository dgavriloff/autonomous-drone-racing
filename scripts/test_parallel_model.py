#!/usr/bin/env python3
"""Test the parallel-trained velocity control model."""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scripts.train_parallel import VelocityRacingEnv, create_simple_track


def test_model(model_path="models/parallel_vel/final", num_episodes=5, gui=True, use_vecnorm=False):
    """Test the trained model."""
    print("=" * 60)
    print("TESTING PARALLEL-TRAINED MODEL")
    print("=" * 60)

    # Create environment
    track = create_simple_track(num_gates=5, radius=1.5)

    if use_vecnorm:
        # Wrap in VecEnv and load normalization stats
        env = DummyVecEnv([lambda: VelocityRacingEnv(track, gui=gui, max_steps=500)])
        # Always use vecnormalize.pkl from the model directory
        vecnorm_path = str(Path(model_path).parent / "vecnormalize.pkl")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False  # Don't update stats during eval
        env.norm_reward = False
        print(f"Loaded VecNormalize from {vecnorm_path}")
    else:
        env = VelocityRacingEnv(track, gui=gui, max_steps=500)

    # Load model
    model = SAC.load(model_path)
    print(f"Loaded model from {model_path}")
    print()

    # Run episodes
    results = []
    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle (obs, info) tuple from non-vec env
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)

            # Handle both VecEnv (obs, reward, done, info) and regular (obs, reward, term, trunc, info)
            if len(result) == 4:  # VecEnv
                obs, reward, dones, infos = result
                done = dones[0] if isinstance(dones, np.ndarray) else dones
                info = infos[0] if isinstance(infos, list) else infos
                reward = reward[0] if isinstance(reward, np.ndarray) else reward
            else:  # Regular env
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated

            total_reward += reward
            steps += 1

            if gui and not use_vecnorm:
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
    parser.add_argument("--vecnorm", action="store_true", help="Use VecNormalize (for models trained with it)")
    args = parser.parse_args()

    test_model(args.model, args.episodes, gui=not args.no_gui, use_vecnorm=args.vecnorm)
