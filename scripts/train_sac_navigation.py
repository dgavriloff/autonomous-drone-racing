#!/usr/bin/env python3
"""
Train SAC agent for gate navigation with PBRS reward.

SAC (Soft Actor-Critic) is used instead of PPO because:
1. Better exploration via entropy regularization
2. More sample efficient
3. Better at escaping local optima (like flying straight up)

The PBRS reward function in high_freq_racing.py provides:
- Directional velocity bonus (only towards gate)
- Distance penalty (prevents flying away)
- Potential-based shaping (policy-invariant guidance)

Usage:
    python scripts/train_sac_navigation.py --timesteps 300000
    python scripts/train_sac_navigation.py --test
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.envs.high_freq_racing import HighFreqRacingAviary, create_monorace_track


class NavigationCallback(BaseCallback):
    """Custom callback to track gate navigation progress."""

    def __init__(self, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_gates = 0
        self.episode_gates = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            gates = info.get("gates_passed", 0)
            self.episode_gates.append(gates)

            if gates > self.best_gates:
                self.best_gates = gates
                if self.verbose:
                    print(f"\n*** NEW BEST: {gates} gates passed! ***")

        # Periodic evaluation
        if self.num_timesteps % self.eval_freq == 0:
            if len(self.episode_gates) > 0:
                avg_gates = np.mean(self.episode_gates[-100:])
                max_gates = max(self.episode_gates[-100:]) if self.episode_gates else 0
                print(f"\n[{self.num_timesteps}] Avg gates: {avg_gates:.2f}, Max: {max_gates}, Best ever: {self.best_gates}")

        return True


def create_env(num_gates=5, gui=False):
    """Create racing environment with PBRS reward."""
    track = create_monorace_track(num_gates=num_gates)

    env = HighFreqRacingAviary(
        track=track,
        ctrl_freq=500,
        pyb_freq=2000,
        gui=gui,
        # Reward tuned for navigation
        reward_gate_passed=100.0,
        reward_velocity_bonus=0.1,  # Now directional only
        reward_progress_bonus=1.0,
        reward_time_penalty=-0.01,
        reward_crash_penalty=-100.0,
        max_episode_steps=1000,
        target_velocity=5.0,
    )

    return Monitor(env)


def train(timesteps=300000, num_gates=5, save_path="models/sac_navigation"):
    """Train SAC agent."""
    print("=" * 60)
    print("SAC Navigation Training")
    print("=" * 60)
    print(f"Timesteps: {timesteps}")
    print(f"Gates: {num_gates}")
    print(f"Save path: {save_path}")
    print()

    # Create environment
    env = create_env(num_gates=num_gates, gui=False)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Create SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",  # Auto-tune entropy
        target_entropy="auto",
        verbose=1,
        tensorboard_log="./logs/sac_navigation",
    )

    # Callbacks
    nav_callback = NavigationCallback(eval_freq=10000)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_path,
        name_prefix="sac_nav",
    )

    # Train
    print("Starting training...")
    start_time = time.time()

    model.learn(
        total_timesteps=timesteps,
        callback=[nav_callback, checkpoint_callback],
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    # Save final model
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.save(f"{save_path}/final_model")
    print(f"Model saved to {save_path}/final_model")

    env.close()

    return model, nav_callback.best_gates


def test(model_path="models/sac_navigation/final_model", num_episodes=10, num_gates=5, gui=False):
    """Test trained SAC agent."""
    print("=" * 60)
    print("Testing SAC Navigation Agent")
    print("=" * 60)

    # Load model
    model = SAC.load(model_path)
    print(f"Loaded model from {model_path}")

    # Create environment
    env = create_env(num_gates=num_gates, gui=gui)

    results = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        trajectory = []  # Track positions

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            trajectory.append(info["position"].copy())

            if terminated or truncated:
                break

        gates_passed = info["gates_passed"]
        avg_speed = info["speed"]

        # Analyze trajectory
        trajectory = np.array(trajectory)
        horizontal_dist = np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2])
        vertical_dist = abs(trajectory[-1, 2] - trajectory[0, 2])

        results.append({
            "gates": gates_passed,
            "reward": total_reward,
            "steps": steps,
            "speed": avg_speed,
            "horizontal_travel": horizontal_dist,
            "vertical_travel": vertical_dist,
        })

        direction = "HORIZONTAL" if horizontal_dist > vertical_dist else "VERTICAL"
        print(f"Episode {ep+1}: {gates_passed}/{num_gates} gates, "
              f"reward={total_reward:.1f}, {direction} (h={horizontal_dist:.2f}, v={vertical_dist:.2f})")

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    avg_gates = np.mean([r["gates"] for r in results])
    max_gates = max([r["gates"] for r in results])
    avg_reward = np.mean([r["reward"] for r in results])
    avg_horizontal = np.mean([r["horizontal_travel"] for r in results])
    avg_vertical = np.mean([r["vertical_travel"] for r in results])

    print(f"Average gates passed: {avg_gates:.2f}/{num_gates}")
    print(f"Best gates passed: {max_gates}/{num_gates}")
    print(f"Average reward: {avg_reward:.1f}")
    print(f"Average horizontal travel: {avg_horizontal:.2f}m")
    print(f"Average vertical travel: {avg_vertical:.2f}m")

    if avg_horizontal > avg_vertical:
        print("\n>>> Agent is navigating HORIZONTALLY (good!)")
    else:
        print("\n>>> Agent still flying UP (needs more work)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train SAC for gate navigation")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--test", action="store_true", help="Test the agent")
    parser.add_argument("--timesteps", type=int, default=300000, help="Training timesteps")
    parser.add_argument("--num_gates", type=int, default=5, help="Number of gates")
    parser.add_argument("--model_path", type=str, default="models/sac_navigation/final_model")
    parser.add_argument("--gui", action="store_true", help="Show GUI during test")
    parser.add_argument("--episodes", type=int, default=10, help="Test episodes")

    args = parser.parse_args()

    if args.train:
        model, best_gates = train(
            timesteps=args.timesteps,
            num_gates=args.num_gates,
        )
        print(f"\nBest gates achieved during training: {best_gates}")

        # Auto-test after training
        print("\n" + "=" * 60)
        print("Auto-testing trained model...")
        test(num_gates=args.num_gates, num_episodes=5)

    elif args.test:
        test(
            model_path=args.model_path,
            num_episodes=args.episodes,
            num_gates=args.num_gates,
            gui=args.gui,
        )
    else:
        # Default: train and test
        print("No action specified. Running training with defaults...")
        train(timesteps=args.timesteps, num_gates=args.num_gates)
        test(num_gates=args.num_gates, num_episodes=5)


if __name__ == "__main__":
    main()
