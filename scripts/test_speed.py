#!/usr/bin/env python3
"""Test and compare speed of trained models."""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC, PPO
from scripts.train_speed import SpeedRacingEnv, create_simple_track


def test_model(
    model_path: str,
    num_episodes: int = 10,
    max_steps: int = 1000,
    gate_tolerance: float = 0.5,
    gui: bool = False,
):
    """Test a model and report speed metrics."""
    print(f"\nTesting: {model_path}")
    print("-" * 50)

    track = create_simple_track(num_gates=5, radius=1.5)
    env = SpeedRacingEnv(
        track,
        gui=gui,
        max_steps=max_steps,
        gate_tolerance=gate_tolerance,
    )

    # Load model
    try:
        model = PPO.load(model_path)
    except:
        model = SAC.load(model_path)

    results = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_speeds = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_speeds.append(info['speed'])

            if gui:
                time.sleep(1/48)

        gates = info['gates_passed']
        steps = info['steps']
        avg_speed = np.mean(ep_speeds)
        max_speed = np.max(ep_speeds)

        results.append({
            'gates': gates,
            'steps': steps,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
        })

        print(f"  Ep {ep+1}: {gates}/5 gates, "
              f"avg={avg_speed:.2f} m/s, max={max_speed:.2f} m/s, "
              f"steps={steps}")

    env.close()

    # Summary
    avg_gates = np.mean([r['gates'] for r in results])
    avg_speed = np.mean([r['avg_speed'] for r in results])
    max_speed = np.max([r['max_speed'] for r in results])
    full_laps = sum(1 for r in results if r['gates'] >= 5)
    avg_steps = np.mean([r['steps'] for r in results if r['gates'] >= 5]) if full_laps > 0 else float('inf')

    print(f"\nSummary:")
    print(f"  Gates: {avg_gates:.2f}/5 ({full_laps}/{num_episodes} full laps)")
    print(f"  Avg speed: {avg_speed:.2f} m/s")
    print(f"  Max speed: {max_speed:.2f} m/s")
    print(f"  Avg lap time: {avg_steps:.0f} steps ({avg_steps/48:.1f}s)")

    return {
        'model': model_path,
        'avg_gates': avg_gates,
        'full_laps': full_laps,
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'avg_lap_steps': avg_steps,
    }


def compare_models(models: list, **kwargs):
    """Compare multiple models."""
    print("=" * 60)
    print("SPEED COMPARISON")
    print("=" * 60)

    results = []
    for model_path in models:
        if Path(model_path).exists() or Path(model_path + ".zip").exists():
            r = test_model(model_path, **kwargs)
            results.append(r)
        else:
            print(f"\nSkipping {model_path} (not found)")

    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':<35} {'Gates':<8} {'Speed':<10} {'Lap Time'}")
    print("-" * 60)
    for r in results:
        name = Path(r['model']).stem
        print(f"{name:<35} {r['avg_gates']:.1f}/5    "
              f"{r['avg_speed']:.2f} m/s   "
              f"{r['avg_lap_steps']:.0f} steps")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "models/curriculum_final.zip",
        "models/speed/final",
    ])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    compare_models(
        args.models,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        gate_tolerance=args.tolerance,
        gui=args.gui,
    )
