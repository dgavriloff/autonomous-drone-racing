#!/usr/bin/env python3
"""
Curriculum Learning with TIGHT tolerance but EASY geometry.

Philosophy: Don't reward imprecision. Simplify the environment instead.

Curriculum stages (all with TIGHT 0.5m tolerance):
1. Small radius (1.0m), 3 gates - gates are close together
2. Small radius (1.0m), 5 gates - full lap but gates still close
3. Medium radius (1.25m), 5 gates - gates further apart
4. Full radius (1.5m), 5 gates - target course

The agent learns precision from the start. We make the COURSE easier, not the
success criteria looser.
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


# Curriculum stages: (radius, num_gates, timesteps_for_stage)
# First verify geometry curriculum works, then add speed
CURRICULUM = [
    (1.0, 3, 300000),   # Stage 1: tiny course, 3 gates - easy to complete lap
    (1.0, 5, 400000),   # Stage 2: tiny course, 5 gates - full lap, still close
    (1.25, 5, 400000),  # Stage 3: medium course - gates spread out more
    (1.5, 5, 500000),   # Stage 4: full course - target difficulty
]

# TIGHT tolerance - we want precision from the start
GATE_TOLERANCE = 0.5  # Half of what we used before


def make_env(rank, seed, num_gates, radius, max_steps, gate_tolerance):
    """Create a single environment instance."""
    def _init():
        track = create_simple_track(num_gates, radius)
        env = VelocityRacingEnv(
            track,
            gui=False,
            max_steps=max_steps,
            gate_tolerance=gate_tolerance,
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


def train_curriculum(n_envs=16, max_steps=1000):
    """Train with curriculum: tight tolerance, easy geometry."""
    print("=" * 60)
    print("CURRICULUM TRAINING")
    print("Tight tolerance (0.5m), easy geometry -> hard geometry")
    print("=" * 60)
    print(f"Parallel envs: {n_envs}")
    print(f"Max steps: {max_steps}")
    print()
    print("Curriculum stages:")
    total_steps = 0
    for i, (radius, gates, steps) in enumerate(CURRICULUM):
        total_steps += steps
        print(f"  Stage {i+1}: radius={radius}m, gates={gates}, steps={steps:,}")
    print(f"  Total: {total_steps:,} steps")
    print()

    model = None
    total_start = time.time()

    for stage_idx, (radius, num_gates, timesteps) in enumerate(CURRICULUM):
        stage = stage_idx + 1
        print()
        print("=" * 60)
        print(f"STAGE {stage}: radius={radius}m, gates={num_gates}, tolerance=0.5m")
        print("=" * 60)

        # Create environments for this stage
        env = SubprocVecEnv([
            make_env(i, 42 + stage * 100, num_gates, radius, max_steps, 0.5)  # Fixed 0.5m tolerance
            for i in range(n_envs)
        ])
        env = VecMonitor(env)

        if model is None:
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
            reset_num_timesteps=(stage == 1),  # Only reset on first stage
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


def test_model(model_path="models/curriculum/final", num_episodes=10, gate_tolerance=0.5):
    """Test on target course (radius=1.5m, 5 gates)."""
    print("=" * 60)
    print(f"Testing on target course (radius=1.5m, 5 gates, tolerance={gate_tolerance}m)")
    print("=" * 60)

    track = create_simple_track(num_gates=5, radius=1.5)
    env = VelocityRacingEnv(track, gui=False, max_steps=1000, gate_tolerance=gate_tolerance)
    model = SAC.load(model_path)

    results = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc

        gates = info["gates_passed"]
        results.append(gates)
        print(f"Episode {ep+1}: {gates}/5 gates, reward={total_reward:.1f}")

    env.close()

    print()
    print(f"Average: {np.mean(results):.2f}/5 gates")
    print(f"Max: {max(results)}/5 gates")
    print(f"Full laps: {sum(1 for g in results if g == 5)}/{num_episodes}")

    return results


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=0.5, help="Gate tolerance (default: 0.5m TIGHT)")
    parser.add_argument("--test", type=str, default=None, help="Path to model to test")
    args = parser.parse_args()

    if args.test:
        test_model(args.test, gate_tolerance=args.tolerance)
    else:
        model = train_curriculum(
            n_envs=args.envs,
            max_steps=args.max_steps,
        )
        print("\nAuto-testing final model...")
        test_model("models/curriculum/final", gate_tolerance=0.5)


if __name__ == "__main__":
    main()
