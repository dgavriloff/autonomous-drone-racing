#!/usr/bin/env python3
"""
Run the vision-based racing pipeline.

Usage:
    python scripts/run_pipeline.py --gui  # With visualization
    python scripts/run_pipeline.py --episodes 10  # Benchmark
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time

from src.pipeline.vision_racing import VisionRacingPipeline, PipelineConfig, PipelineRunner
from src.envs.high_freq_racing import HighFreqRacingAviary, create_monorace_track


def parse_args():
    parser = argparse.ArgumentParser(description="Run vision-based racing pipeline")

    # Environment
    parser.add_argument("--gui", action="store_true",
                        help="Show PyBullet GUI")
    parser.add_argument("--num_gates", type=int, default=5,
                        help="Number of gates in track")

    # Pipeline
    parser.add_argument("--control_freq", type=int, default=500,
                        help="Control frequency (Hz)")
    parser.add_argument("--vision_freq", type=int, default=24,
                        help="Vision frequency (Hz)")
    parser.add_argument("--target_velocity", type=float, default=5.0,
                        help="Target velocity (m/s)")

    # Models
    parser.add_argument("--gate_net", type=str, default=None,
                        help="Path to GateNet checkpoint")
    parser.add_argument("--gcnet", type=str, default=None,
                        help="Path to G&CNet checkpoint")

    # Episode
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode")

    # Output
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed info")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Vision-Based Racing Pipeline")
    print("=" * 60)

    # Create track
    print(f"\nCreating track with {args.num_gates} gates...")
    track = create_monorace_track(num_gates=args.num_gates)

    # Create known gates dict for EKF
    known_gates = {}
    for i, gate in enumerate(track.gates):
        # Convert PyBullet quaternion (x,y,z,w) to EKF format (w,x,y,z)
        quat_wxyz = np.array([
            gate.orientation[3],
            gate.orientation[0],
            gate.orientation[1],
            gate.orientation[2],
        ])
        known_gates[i] = (gate.position, quat_wxyz)

    # Create environment
    print(f"Creating environment (GUI: {args.gui})...")
    env = HighFreqRacingAviary(
        track=track,
        ctrl_freq=args.control_freq,
        pyb_freq=args.control_freq * 4,  # 4x control freq
        gui=args.gui,
    )

    # Create pipeline
    print("Creating pipeline...")
    config = PipelineConfig(
        control_freq=args.control_freq,
        vision_freq=args.vision_freq,
        target_velocity=args.target_velocity,
        gate_net_path=args.gate_net,
        gcnet_path=args.gcnet,
        device=args.device,
    )

    pipeline = VisionRacingPipeline(config=config, known_gates=known_gates)

    print(f"  Control freq: {config.control_freq} Hz")
    print(f"  Vision freq: {config.vision_freq} Hz")
    print(f"  Target velocity: {config.target_velocity} m/s")
    print(f"  Device: {pipeline.device}")

    # Create runner
    runner = PipelineRunner(pipeline, env)

    # Run episodes
    print(f"\nRunning {args.episodes} episode(s)...")

    all_results = []

    for ep in range(args.episodes):
        if args.verbose:
            print(f"\n--- Episode {ep + 1}/{args.episodes} ---")

        start_time = time.time()
        results = runner.run_episode(max_steps=args.max_steps)
        elapsed = time.time() - start_time

        results["elapsed_time"] = elapsed
        all_results.append(results)

        if args.verbose:
            print(f"  Steps: {results['steps']}")
            print(f"  Gates passed: {results['gates_passed']}/{args.num_gates}")
            print(f"  Total reward: {results['total_reward']:.2f}")
            print(f"  Avg speed: {results['avg_speed']:.2f} m/s")
            print(f"  Max speed: {results['max_speed']:.2f} m/s")
            print(f"  Elapsed: {elapsed:.2f}s")

    env.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    avg_reward = np.mean([r["total_reward"] for r in all_results])
    avg_gates = np.mean([r["gates_passed"] for r in all_results])
    avg_speed = np.mean([r["avg_speed"] for r in all_results])
    avg_steps = np.mean([r["steps"] for r in all_results])

    print(f"Episodes: {args.episodes}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average gates passed: {avg_gates:.2f}/{args.num_gates}")
    print(f"Average speed: {avg_speed:.2f} m/s")
    print(f"Average steps: {avg_steps:.0f}")

    # Success rate
    success_rate = np.mean([r["gates_passed"] >= args.num_gates for r in all_results])
    print(f"Success rate: {success_rate * 100:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
