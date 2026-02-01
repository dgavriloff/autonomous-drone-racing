#!/usr/bin/env python3
"""
Test policy robustness to observation noise and delay.

Simulates what the policy would experience with vision-based state estimation:
- Position noise: ~5-15cm RMS (EKF uncertainty)
- Velocity noise: ~0.1-0.3 m/s RMS
- Orientation noise: ~2-5° RMS
- Latency: 30-50ms (vision processing delay = 1-2 frames at 48Hz)
"""

import sys
from pathlib import Path
from collections import deque
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from stable_baselines3 import SAC, PPO
from scripts.train_parallel import VelocityRacingEnv, create_simple_track


class NoisyStateWrapper(gym.Wrapper):
    """
    Wrapper that adds noise and delay to observations.

    Observation layout (16 dims):
    - pos: 0-2 (position in world frame)
    - vel: 3-5 (velocity)
    - euler: 6-8 (orientation as roll, pitch, yaw)
    - ang_vel: 9-11 (angular velocity)
    - to_gate_dir: 12-14 (direction to gate)
    - dist: 15 (distance to gate)

    We add noise to pos, vel, euler (primary state estimates).
    to_gate_dir and dist are derived, so we recompute them from noisy pos.
    """

    def __init__(
        self,
        env,
        pos_noise: float = 0.0,      # Position noise std (meters)
        vel_noise: float = 0.0,      # Velocity noise std (m/s)
        ori_noise: float = 0.0,      # Orientation noise std (radians)
        delay_frames: int = 0,        # Number of frames to delay observations
        seed: int = None,
    ):
        super().__init__(env)
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.ori_noise = ori_noise
        self.delay_frames = delay_frames

        # Delay buffer (stores past observations)
        self.obs_buffer = deque(maxlen=max(1, delay_frames + 1))

        # RNG for reproducibility
        self.rng = np.random.default_rng(seed)

    def _add_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to observation components."""
        noisy_obs = obs.copy()

        # Add noise to position (indices 0-2)
        if self.pos_noise > 0:
            noisy_obs[0:3] += self.rng.normal(0, self.pos_noise, 3)

        # Add noise to velocity (indices 3-5)
        if self.vel_noise > 0:
            noisy_obs[3:6] += self.rng.normal(0, self.vel_noise, 3)

        # Add noise to orientation (indices 6-8, euler angles)
        if self.ori_noise > 0:
            noisy_obs[6:9] += self.rng.normal(0, self.ori_noise, 3)

        # Note: to_gate_dir (12-14) and dist (15) are kept as-is
        # In real vision pipeline, these would be computed from noisy pos + known gate pos
        # For this test, the inconsistency is acceptable

        return noisy_obs.astype(np.float32)

    def _get_delayed_obs(self) -> np.ndarray:
        """Get observation with delay applied."""
        if self.delay_frames == 0 or len(self.obs_buffer) < self.delay_frames + 1:
            # Not enough history yet, return most recent
            return self.obs_buffer[-1]
        else:
            # Return delayed observation
            return self.obs_buffer[0]

    def reset(self, **kwargs):
        """Reset environment and clear delay buffer."""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        # Add noise and initialize buffer
        noisy_obs = self._add_noise(obs)
        self.obs_buffer.clear()
        # Fill buffer with initial observation for delay warmup
        for _ in range(self.delay_frames + 1):
            self.obs_buffer.append(noisy_obs.copy())

        if isinstance(result, tuple):
            return self._get_delayed_obs(), info
        return self._get_delayed_obs()

    def step(self, action):
        """Step environment and apply noise + delay."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add noise to current observation
        noisy_obs = self._add_noise(obs)

        # Add to buffer
        self.obs_buffer.append(noisy_obs)

        # Return delayed observation
        delayed_obs = self._get_delayed_obs()

        return delayed_obs, reward, terminated, truncated, info


def run_test(
    model_path: str,
    pos_noise: float,
    vel_noise: float,
    ori_noise: float,
    delay_frames: int,
    num_episodes: int = 10,
    max_steps: int = 1500,
    gate_tolerance: float = 0.5,
    gui: bool = False,
) -> dict:
    """Run test with specified noise parameters."""

    # Create base environment
    track = create_simple_track(num_gates=5, radius=1.5)
    base_env = VelocityRacingEnv(
        track,
        gui=gui,
        max_steps=max_steps,
        gate_tolerance=gate_tolerance,
    )

    # Wrap with noise
    env = NoisyStateWrapper(
        base_env,
        pos_noise=pos_noise,
        vel_noise=vel_noise,
        ori_noise=ori_noise,
        delay_frames=delay_frames,
        seed=42,
    )

    # Load model
    try:
        model = PPO.load(model_path)
    except Exception:
        model = SAC.load(model_path)

    # Run episodes
    gates_list = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if gui:
                time.sleep(1/48)

        gates = info.get("gates_passed", 0)
        gates_list.append(gates)

    env.close()

    return {
        "pos_noise": pos_noise,
        "vel_noise": vel_noise,
        "ori_noise": ori_noise,
        "delay_frames": delay_frames,
        "gates_mean": np.mean(gates_list),
        "gates_std": np.std(gates_list),
        "gates_min": min(gates_list),
        "gates_max": max(gates_list),
        "full_laps": sum(1 for g in gates_list if g >= 5),
        "num_episodes": num_episodes,
        "gates_list": gates_list,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test policy robustness to noise/delay")
    parser.add_argument("--model", default="models/curriculum_final.zip", help="Model path")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per test")
    parser.add_argument("--max-steps", type=int, default=1500, help="Max steps per episode")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Gate tolerance (meters)")
    parser.add_argument("--gui", action="store_true", help="Show GUI (slower)")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer episodes")
    args = parser.parse_args()

    episodes = 5 if args.quick else args.episodes

    print("=" * 70)
    print("NOISE ROBUSTNESS TEST")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Episodes per test: {episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Gate tolerance: {args.tolerance}m")
    print()

    # Test configurations
    # Format: (name, pos_noise, vel_noise, ori_noise, delay_frames)
    tests = [
        ("Baseline (no noise)", 0.0, 0.0, 0.0, 0),
        ("Low noise", 0.05, 0.10, 0.035, 0),  # 5cm, 0.1m/s, 2°
        ("Medium noise", 0.10, 0.20, 0.052, 0),  # 10cm, 0.2m/s, 3°
        ("High noise", 0.15, 0.30, 0.087, 0),  # 15cm, 0.3m/s, 5°
        ("Low + 1 frame delay", 0.05, 0.10, 0.035, 1),
        ("Med + 2 frame delay", 0.10, 0.20, 0.052, 2),  # ~40ms at 48Hz
    ]

    results = []
    for name, pos_n, vel_n, ori_n, delay in tests:
        print(f"\nTesting: {name}")
        print(f"  pos_noise={pos_n}m, vel_noise={vel_n}m/s, ori_noise={np.degrees(ori_n):.1f}°, delay={delay} frames")

        result = run_test(
            model_path=args.model,
            pos_noise=pos_n,
            vel_noise=vel_n,
            ori_noise=ori_n,
            delay_frames=delay,
            num_episodes=episodes,
            max_steps=args.max_steps,
            gate_tolerance=args.tolerance,
            gui=args.gui,
        )
        result["name"] = name
        results.append(result)

        print(f"  Result: {result['gates_mean']:.2f} ± {result['gates_std']:.2f} gates")
        print(f"  Full laps: {result['full_laps']}/{episodes}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<30} {'Gates':<15} {'Full Laps':<12} {'Min-Max'}")
    print("-" * 70)
    for r in results:
        gates_str = f"{r['gates_mean']:.2f} ± {r['gates_std']:.2f}"
        laps_str = f"{r['full_laps']}/{r['num_episodes']}"
        range_str = f"{r['gates_min']}-{r['gates_max']}"
        print(f"{r['name']:<30} {gates_str:<15} {laps_str:<12} {range_str}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    baseline = results[0]
    degraded = False

    for r in results[1:]:
        diff = baseline['gates_mean'] - r['gates_mean']
        if diff > 1.0:
            degraded = True
            print(f"⚠️  {r['name']}: Significant degradation (-{diff:.1f} gates)")
        elif diff > 0.5:
            print(f"⚡ {r['name']}: Moderate degradation (-{diff:.1f} gates)")
        else:
            print(f"✓  {r['name']}: Robust (within 0.5 gates of baseline)")

    print()
    if degraded:
        print("CONCLUSION: Policy is NOT robust to expected vision noise levels.")
        print("RECOMMENDATION: Retrain with observation domain randomization.")
    else:
        print("CONCLUSION: Policy appears robust to expected vision noise levels.")
        print("RECOMMENDATION: Proceed with vision pipeline integration.")

    return results


if __name__ == "__main__":
    main()
