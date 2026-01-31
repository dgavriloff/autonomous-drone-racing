#!/usr/bin/env python3
"""
Curriculum Learning for Gate Navigation.

The problem: Agent struggles to learn navigation when gates are 3m away.
Solution: Start with gates 0.5m away, progressively increase difficulty.

Curriculum stages:
1. Stage 1: 1 gate, 0.5m away (learn basic movement toward target)
2. Stage 2: 2 gates, 1.0m radius (learn sequential navigation)
3. Stage 3: 3 gates, 1.5m radius (more gates)
4. Stage 4: 5 gates, 2.0m radius (near-full track)
5. Stage 5: 5 gates, 3.0m radius (full difficulty)

Progression: Move to next stage when avg gates > 80% for 50 episodes.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import pybullet as p
from src.envs.high_freq_racing import HighFreqRacingAviary, TrackConfig, GateConfig


def create_curriculum_track(num_gates: int, radius: float) -> TrackConfig:
    """Create a track with specified gates and radius."""
    gates = []
    base_height = 0.5  # Lower height for easier learning

    for i in range(num_gates):
        angle = 2 * np.pi * i / num_gates
        next_angle = 2 * np.pi * (i + 1) / num_gates

        # Position on circle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = base_height

        # Gate faces towards next gate
        direction = np.array([
            np.cos(next_angle) - np.cos(angle),
            np.sin(next_angle) - np.sin(angle),
            0,
        ])
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        yaw = np.arctan2(direction[1], direction[0])
        quat = p.getQuaternionFromEuler([0, 0, yaw])

        gates.append(GateConfig(
            position=np.array([x, y, z]),
            orientation=np.array(quat),
        ))

    # Start position: AT the first gate (inside tolerance)
    start_x = radius * 0.95  # Very close to first gate
    start_y = 0.0
    start_z = base_height

    return TrackConfig(
        name=f"curriculum_{num_gates}g_{radius}m",
        gates=gates,
        start_position=np.array([start_x, start_y, start_z]),
    )


class CurriculumCallback(BaseCallback):
    """Callback that handles curriculum progression."""

    STAGES = [
        # Gradual progression - teach movement first
        {"num_gates": 1, "radius": 0.3, "name": "Stage 1: 1 gate, 0.3m (pass immediately)"},
        {"num_gates": 2, "radius": 0.3, "name": "Stage 2: 2 gates, 0.3m (0.4m between gates)"},
        {"num_gates": 2, "radius": 0.5, "name": "Stage 3: 2 gates, 0.5m (0.7m between gates)"},
        {"num_gates": 3, "radius": 0.5, "name": "Stage 4: 3 gates, 0.5m"},
        {"num_gates": 3, "radius": 0.8, "name": "Stage 5: 3 gates, 0.8m"},
        {"num_gates": 5, "radius": 1.0, "name": "Stage 6: 5 gates, 1.0m"},
        {"num_gates": 5, "radius": 2.0, "name": "Stage 7: 5 gates, 2.0m"},
        {"num_gates": 5, "radius": 3.0, "name": "Stage 8: 5 gates, 3.0m (full)"},
    ]

    def __init__(self, verbose=1, progress_threshold=0.8, window_size=50):
        super().__init__(verbose)
        self.current_stage = 0
        self.progress_threshold = progress_threshold
        self.window_size = window_size
        self.episode_gates = []
        self.stage_start_timesteps = 0
        self.best_gates_ever = 0

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            gates = info.get("gates_passed", 0)
            max_gates = self.STAGES[self.current_stage]["num_gates"]
            self.episode_gates.append(gates / max_gates)  # Normalized

            if gates > self.best_gates_ever:
                self.best_gates_ever = gates
                if self.verbose:
                    print(f"\n*** NEW BEST: {gates} gates! ***")

            # Check progression
            if len(self.episode_gates) >= self.window_size:
                recent_avg = np.mean(self.episode_gates[-self.window_size:])

                if recent_avg >= self.progress_threshold:
                    if self.current_stage < len(self.STAGES) - 1:
                        self._advance_stage()

        # Periodic status
        if self.num_timesteps % 10000 == 0:
            stage = self.STAGES[self.current_stage]
            if len(self.episode_gates) > 0:
                recent_avg = np.mean(self.episode_gates[-min(50, len(self.episode_gates)):])
                print(f"\n[{self.num_timesteps}] {stage['name']} | "
                      f"Avg completion: {recent_avg*100:.1f}% | Best: {self.best_gates_ever}")

        return True

    def _advance_stage(self):
        """Advance to next curriculum stage."""
        self.current_stage += 1
        stage = self.STAGES[self.current_stage]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ADVANCING TO {stage['name']}")
            print(f"{'='*60}\n")

        # Reset tracking for new stage
        self.episode_gates = []
        self.stage_start_timesteps = self.num_timesteps

        # Update environment
        track = create_curriculum_track(stage["num_gates"], stage["radius"])
        self.training_env.envs[0].track = track
        self.training_env.envs[0].reset()


def create_env(num_gates=1, radius=0.5, gui=False):
    """Create curriculum environment with tuned rewards."""
    track = create_curriculum_track(num_gates, radius)

    env = HighFreqRacingAviary(
        track=track,
        ctrl_freq=500,
        pyb_freq=2000,
        gui=gui,
        # Tuned rewards for curriculum learning
        reward_gate_passed=200.0,  # Big bonus for gates
        reward_velocity_bonus=0.5,  # Encourage movement towards gate
        reward_progress_bonus=5.0,  # Strong PBRS signal
        reward_time_penalty=-0.005,  # Smaller time penalty
        reward_crash_penalty=-20.0,  # Reduced crash penalty
        max_episode_steps=300,  # Shorter episodes
        target_velocity=2.0,  # Lower target velocity
        gate_tolerance=0.25,  # Gate detection radius
    )

    return Monitor(env)


def train(timesteps=500000):
    """Train with curriculum learning."""
    print("=" * 60)
    print("CURRICULUM LEARNING FOR GATE NAVIGATION")
    print("=" * 60)
    print()
    print("Curriculum stages:")
    for i, stage in enumerate(CurriculumCallback.STAGES):
        print(f"  {i+1}. {stage['name']}")
    print()
    print(f"Progression: Move to next when >80% gates for 50 episodes")
    print(f"Total timesteps: {timesteps}")
    print()

    # Start with Stage 1
    stage = CurriculumCallback.STAGES[0]
    env = create_env(stage["num_gates"], stage["radius"])

    print(f"Starting with {stage['name']}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Create SAC
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=500,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        verbose=1,
        tensorboard_log="./logs/curriculum",
    )

    # Curriculum callback
    curriculum_cb = CurriculumCallback(verbose=1)

    # Train
    print("Starting training...")
    start_time = time.time()

    model.learn(
        total_timesteps=timesteps,
        callback=curriculum_cb,
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Final stage: {CurriculumCallback.STAGES[curriculum_cb.current_stage]['name']}")
    print(f"Best gates ever: {curriculum_cb.best_gates_ever}")

    # Save
    Path("models/curriculum").mkdir(parents=True, exist_ok=True)
    model.save("models/curriculum/final_model")
    print("Model saved to models/curriculum/final_model")

    env.close()

    return model, curriculum_cb


def test(model_path="models/curriculum/final_model", num_episodes=10):
    """Test on full difficulty track."""
    print("=" * 60)
    print("Testing on full difficulty (5 gates, 3m radius)")
    print("=" * 60)

    model = SAC.load(model_path)
    env = create_env(num_gates=5, radius=3.0)

    results = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        trajectory = []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            trajectory.append(info["position"].copy())

            if terminated or truncated:
                break

        gates = info["gates_passed"]
        traj = np.array(trajectory)
        h_dist = np.linalg.norm(traj[-1, :2] - traj[0, :2]) if len(traj) > 1 else 0
        v_dist = abs(traj[-1, 2] - traj[0, 2]) if len(traj) > 1 else 0

        results.append({"gates": gates, "reward": total_reward, "h": h_dist, "v": v_dist})
        direction = "HORIZONTAL" if h_dist > v_dist else "VERTICAL"
        print(f"Episode {ep+1}: {gates}/5 gates, reward={total_reward:.1f}, {direction}")

    env.close()

    avg_gates = np.mean([r["gates"] for r in results])
    print(f"\nAverage gates: {avg_gates:.2f}/5")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--timesteps", type=int, default=500000)
    args = parser.parse_args()

    if args.train:
        model, cb = train(timesteps=args.timesteps)
        print("\nAuto-testing...")
        test()
    elif args.test:
        test()
    else:
        train(timesteps=args.timesteps)
        test()


if __name__ == "__main__":
    main()
