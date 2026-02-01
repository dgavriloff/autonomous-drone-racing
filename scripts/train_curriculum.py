#!/usr/bin/env python3
"""
Curriculum Learning: RACE Drone (830g, 200 km/h max)

Phase 1 - Geometry (~1.7 m/s baseline):
1. r=1.5m, 3 gates
2. r=1.5m, 5 gates
3. r=2.0m, 5 gates
4. r=2.5m, 5 gates

Phase 2 - Speed with SCALED RADIUS (competition speeds):
5.  2.8 m/s,  r=2m
6.  5.6 m/s,  r=3m
7.  8.3 m/s,  r=4m
8.  11.1 m/s, r=5m
9.  15 m/s,   r=7m
10. 20 m/s,   r=10m  (competition level!)
11. 25 m/s,   r=15m
12. 30 m/s,   r=20m  (Swift/MonoRace level!)

Key insights:
- RACE drone has 200 km/h max (55 m/s), not 30 km/h like Crazyflie
- speed_factor * 200 * (1000/3600) = actual speed in m/s
- Larger radii needed for high-speed turns (a = v²/r)
- Target: 20-30 m/s with reliable gate completion
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from scripts.train_parallel import VelocityRacingEnv, create_simple_track


# Curriculum stages: (radius, num_gates, speed_factor, tolerance, timesteps)
# CF2X drone: MAX_SPEED_KMH = 30 (8.33 m/s max)
# speed_factor * 30 * (1000/3600) = speed in m/s
# 0.1 = 0.83 m/s, 0.3 = 2.5 m/s, 0.5 = 4.17 m/s, 0.8 = 6.67 m/s, 1.0 = 8.33 m/s

# Phase 1: Geometry curriculum at slow speed
GEOMETRY_CURRICULUM = [
    (1.5, 3, 0.1, 0.6, 300000),    # Stage 1: small course, 3 gates, ~0.83 m/s
    (1.5, 5, 0.1, 0.6, 400000),    # Stage 2: small course, 5 gates
    (2.0, 5, 0.1, 0.6, 400000),    # Stage 3: medium course
    (2.5, 5, 0.1, 0.6, 500000),    # Stage 4: larger course - geometry complete
]

# Phase 2: Speed curriculum with SCALED RADIUS (a = v²/r)
# Larger radii needed for higher speeds to keep centripetal acceleration manageable
SPEED_CURRICULUM = [
    (2.0, 5, 0.2, 0.6, 400000),    # Stage 5: 1.67 m/s
    (2.5, 5, 0.3, 0.7, 400000),    # Stage 6: 2.5 m/s
    (3.0, 5, 0.4, 0.8, 500000),    # Stage 7: 3.33 m/s
    (4.0, 5, 0.5, 0.8, 500000),    # Stage 8: 4.17 m/s
    (5.0, 5, 0.6, 0.9, 500000),    # Stage 9: 5.0 m/s
    (6.0, 5, 0.7, 1.0, 500000),    # Stage 10: 5.83 m/s
    (7.0, 5, 0.8, 1.0, 600000),    # Stage 11: 6.67 m/s
    (8.0, 5, 1.0, 1.2, 600000),    # Stage 12: 8.33 m/s (CF2X max!)
]

# Combined curriculum
CURRICULUM = GEOMETRY_CURRICULUM + SPEED_CURRICULUM


def make_env(rank, seed, num_gates, radius, max_steps, gate_tolerance, speed_factor=0.03):
    """Create a single environment instance."""
    def _init():
        track = create_simple_track(num_gates, radius)
        env = VelocityRacingEnv(
            track,
            gui=False,
            max_steps=max_steps,
            gate_tolerance=gate_tolerance,
            speed_factor=speed_factor,
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class CurriculumCallback(BaseCallback):
    """Track progress for curriculum stage."""

    def __init__(self, stage, num_gates, verbose=1):
        super().__init__(verbose)
        self.stage = stage
        self.num_gates = num_gates
        self.best_gates = 0
        self.episode_gates = []
        self.full_laps = 0

    def _on_step(self):
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]
                gates = info.get("gates_passed", 0)
                self.episode_gates.append(gates)

                if gates > self.best_gates:
                    self.best_gates = gates
                    print(f"\n*** Stage {self.stage}: NEW BEST {gates}/{self.num_gates} gates! ***")

                if gates == self.num_gates:
                    self.full_laps += 1

        if self.num_timesteps % 10000 == 0 and self.episode_gates:
            recent = self.episode_gates[-100:]
            avg = np.mean(recent)
            max_recent = max(recent)
            full_lap_rate = sum(1 for g in recent if g == self.num_gates) / len(recent) * 100
            print(f"\n[Stage {self.stage}][{self.num_timesteps}] "
                  f"Avg: {avg:.2f}/{self.num_gates}, Max: {max_recent}, "
                  f"Full laps: {full_lap_rate:.0f}%")

        return True


def train_curriculum(n_envs=16, max_steps=1000, start_stage=1, resume_from=None):
    """Train with curriculum: geometry first, then speed."""
    print("=" * 60)
    print("CURRICULUM TRAINING")
    print("Phase 1: Geometry (0.25 m/s) -> Phase 2: Speed (up to 2.0 m/s)")
    print("=" * 60)
    print(f"Parallel envs: {n_envs}")
    print(f"Max steps: {max_steps}")
    print(f"Starting from stage: {start_stage}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print()
    print("Curriculum stages:")
    total_steps = 0
    for i, (radius, gates, speed_factor, tolerance, steps) in enumerate(CURRICULUM):
        speed_ms = speed_factor * 200 * (1000/3600)  # RACE drone MAX_SPEED_KMH = 200
        total_steps += steps
        phase = "Geometry" if i < len(GEOMETRY_CURRICULUM) else "Speed"
        print(f"  Stage {i+1} [{phase}]: radius={radius}m, gates={gates}, "
              f"speed={speed_ms:.2f}m/s, tol={tolerance}m, steps={steps:,}")
    print(f"  Total: {total_steps:,} steps")
    print()

    model = None
    total_start = time.time()

    for stage_idx, (radius, num_gates, speed_factor, tolerance, timesteps) in enumerate(CURRICULUM):
        stage = stage_idx + 1

        # Skip stages before start_stage
        if stage < start_stage:
            print(f"Skipping stage {stage}...")
            continue

        speed_ms = speed_factor * 200 * (1000/3600)  # RACE drone
        phase = "Geometry" if stage <= len(GEOMETRY_CURRICULUM) else "Speed"

        print()
        print("=" * 60)
        print(f"STAGE {stage} [{phase}]: radius={radius}m, gates={num_gates}, "
              f"speed={speed_ms:.2f}m/s, tol={tolerance}m")
        print("=" * 60)

        # Create environments for this stage
        env = SubprocVecEnv([
            make_env(i, 42 + stage * 100, num_gates, radius, max_steps, tolerance, speed_factor)
            for i in range(n_envs)
        ])
        env = VecMonitor(env)

        if model is None:
            if resume_from:
                # Load pretrained model
                print(f"Loading model from {resume_from}...")
                model = PPO.load(resume_from, env=env, verbose=1, tensorboard_log="./logs/curriculum")
            else:
                # First stage: create new model
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,  # Fixed entropy, no collapse
                    verbose=1,
                    tensorboard_log="./logs/curriculum",
                )
        else:
            # Subsequent stages: transfer to new environment
            model.set_env(env)

        # Callbacks
        curriculum_cb = CurriculumCallback(stage, num_gates)
        checkpoint_cb = CheckpointCallback(
            save_freq=50000,
            save_path="models/curriculum",
            name_prefix=f"stage{stage}",
        )

        # Train this stage
        print(f"Training stage {stage} for {timesteps:,} steps...")
        start = time.time()

        model.learn(
            total_timesteps=timesteps,
            callback=[curriculum_cb, checkpoint_cb],
            progress_bar=True,
            reset_num_timesteps=(stage == start_stage),  # Only reset on first active stage
        )

        elapsed = time.time() - start
        print(f"\nStage {stage} completed in {elapsed/60:.1f} minutes")
        print(f"Best gates: {curriculum_cb.best_gates}/{num_gates}")
        print(f"Full laps completed: {curriculum_cb.full_laps}")

        # Save stage checkpoint
        Path("models/curriculum").mkdir(parents=True, exist_ok=True)
        stage_path = f"models/curriculum/stage{stage}_final"
        model.save(stage_path)
        print(f"Saved to {stage_path}")

        env.close()

    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print(f"CURRICULUM COMPLETE in {total_elapsed/60:.1f} minutes")
    print("=" * 60)

    # Save final model
    model.save("models/curriculum/final")
    print("Final model saved to models/curriculum/final")

    return model


def test_model(model_path="models/curriculum/final", num_episodes=10, gate_tolerance=0.5, speed_factor=0.03):
    """Test on target course."""
    speed_ms = speed_factor * 200 * (1000/3600)  # RACE drone: MAX_SPEED_KMH = 200
    print("=" * 60)
    print(f"Testing on target course (radius=1.5m, 5 gates)")
    print(f"  tolerance={gate_tolerance}m, speed={speed_ms:.2f} m/s")
    print("=" * 60)

    track = create_simple_track(num_gates=5, radius=1.5)
    env = VelocityRacingEnv(track, gui=False, max_steps=1000, gate_tolerance=gate_tolerance, speed_factor=speed_factor)
    model = PPO.load(model_path)

    results = []
    speeds = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        max_speed = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            max_speed = max(max_speed, info.get("speed", 0))
            done = term or trunc

        gates = info["gates_passed"]
        results.append(gates)
        speeds.append(max_speed)
        print(f"Episode {ep+1}: {gates}/5 gates, max_speed={max_speed:.2f}m/s, reward={total_reward:.1f}")

    env.close()

    print()
    print(f"Average: {np.mean(results):.2f}/5 gates")
    print(f"Max: {max(results)}/5 gates")
    print(f"Full laps: {sum(1 for g in results if g == 5)}/{num_episodes}")
    print(f"Average max speed: {np.mean(speeds):.2f} m/s")

    return results


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=0.5, help="Gate tolerance for testing (default: 0.5m)")
    parser.add_argument("--test", type=str, default=None, help="Path to model to test")
    parser.add_argument("--start-stage", type=int, default=1, help="Start from this stage (default: 1)")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume from")
    parser.add_argument("--speed-only", action="store_true", help="Skip geometry, start at speed stage 5")
    parser.add_argument("--speed-factor", type=float, default=0.36, help="Speed factor for testing (default: 0.36 = 20 m/s for RACE drone)")
    args = parser.parse_args()

    if args.test:
        test_model(args.test, gate_tolerance=args.tolerance, speed_factor=args.speed_factor)
    else:
        start_stage = args.start_stage
        if args.speed_only:
            start_stage = len(GEOMETRY_CURRICULUM) + 1  # Start at first speed stage
            print(f"Speed-only mode: starting at stage {start_stage}")

        model = train_curriculum(
            n_envs=args.envs,
            max_steps=args.max_steps,
            start_stage=start_stage,
            resume_from=args.resume,
        )
        print("\nAuto-testing final model at 20 m/s...")
        test_model("models/curriculum/final", gate_tolerance=1.2, speed_factor=0.36)


if __name__ == "__main__":
    main()
