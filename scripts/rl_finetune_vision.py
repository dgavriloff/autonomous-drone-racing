#!/usr/bin/env python3
"""
RL Fine-tuning for vision-based drone racing.

Uses PPO with CNN policy to fine-tune from BC checkpoint.
Key: Uses 16 parallel envs for hardware efficiency.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

from scripts.train_parallel import create_simple_track
from scripts.collect_teacher_demos import CameraRacingEnv


class VisionRacingEnv(gym.Env):
    """Vision-based racing environment for RL."""

    def __init__(self, num_gates=5, image_size=(64, 48), num_frames=4):
        super().__init__()
        self.num_gates = num_gates
        self.image_size = image_size
        self.num_frames = num_frames

        # Create underlying env
        track = create_simple_track(num_gates=num_gates, radius=1.5)
        self.env = CameraRacingEnv(
            track=track,
            image_size=image_size,
            ctrl_freq=48,
            pyb_freq=240,
            gui=False,
            gate_tolerance=0.8,
            max_steps=500,
        )

        # Observation: stacked frames (H, W, C*num_frames) -> (C*num_frames, H, W)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3 * num_frames, image_size[1], image_size[0]),  # (C, H, W)
            dtype=np.float32
        )

        # Action: velocity commands
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        self.frame_buffer = []
        self.prev_gate = 0
        self.prev_dist = None

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        self.frame_buffer = []
        self.prev_gate = 0
        self.prev_dist = None

        # Get initial observation
        rgb = self.env.get_camera_image()
        return self._get_obs(rgb), info

    def _get_obs(self, rgb):
        """Stack frames and return observation."""
        rgb_norm = rgb.astype(np.float32) / 255.0
        self.frame_buffer.append(rgb_norm)
        if len(self.frame_buffer) > self.num_frames:
            self.frame_buffer.pop(0)

        # Pad if needed
        if len(self.frame_buffer) < self.num_frames:
            padded = [self.frame_buffer[0]] * (self.num_frames - len(self.frame_buffer)) + self.frame_buffer
        else:
            padded = self.frame_buffer

        # Stack along channel dimension: (H, W, C*num_frames) -> (C*num_frames, H, W)
        stacked = np.concatenate(padded, axis=-1)  # (H, W, C*num_frames)
        obs = np.transpose(stacked, (2, 0, 1))  # (C*num_frames, H, W)
        return obs

    def step(self, action):
        # Step underlying env
        obs, _, terminated, truncated, info = self.env.step(action.reshape(1, -1))

        # Get camera observation
        rgb = self.env.get_camera_image()
        vision_obs = self._get_obs(rgb)

        # Custom reward
        reward = self._compute_reward(info, obs)

        return vision_obs, reward, terminated, truncated, info

    def _compute_reward(self, info, obs):
        """Reward function optimized for gate completion."""
        reward = 0.0

        # Gate passed bonus
        current_gate = info.get("gates_passed", 0)
        if current_gate > self.prev_gate:
            reward += 100.0  # Big bonus for passing gate
            self.prev_gate = current_gate

        # Progress toward next gate
        if hasattr(self.env, 'track') and hasattr(self.env, 'current_gate_idx'):
            try:
                gate_idx = min(self.env.current_gate_idx, len(self.env.track.gates) - 1)
                gate_pos = np.array(self.env.track.gates[gate_idx].position)
                drone_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
                dist = np.linalg.norm(drone_pos - gate_pos)

                if self.prev_dist is not None:
                    progress = self.prev_dist - dist
                    reward += progress * 10.0  # Reward for getting closer
                self.prev_dist = dist
            except:
                pass

        # Small time penalty
        reward -= 0.1

        # Crash penalty
        if info.get("crashed", False):
            reward -= 50.0

        return reward

    def close(self):
        self.env.close()


class CustomCNN(BaseFeaturesExtractor):
    """CNN feature extractor matching VisionStudentNetV2 architecture."""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 12 for 4 frames

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing a forward pass
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


def make_env(rank, seed=0):
    """Create a single env instance."""
    def _init():
        env = VisionRacingEnv(num_gates=5, num_frames=4)
        env.reset(seed=seed + rank)
        return env
    return _init


class ProgressCallback(BaseCallback):
    """Print progress during training."""

    def __init__(self, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Get recent episode rewards
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_length = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                print(f"Step {self.n_calls}: mean_reward={mean_reward:.1f}, mean_length={mean_length:.0f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"  New best reward: {mean_reward:.1f}")
        return True


def evaluate_gates(model, n_episodes=10):
    """Evaluate how many gates the model passes."""
    env = VisionRacingEnv(num_gates=5, num_frames=4)
    gates_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        gates_list.append(info.get("gates_passed", 0))

    env.close()
    return gates_list


def load_bc_weights_into_ppo(model, bc_checkpoint_path):
    """Load BC weights into PPO's feature extractor."""
    print(f"\nLoading BC weights from: {bc_checkpoint_path}")

    checkpoint = torch.load(bc_checkpoint_path, map_location=model.device, weights_only=False)
    bc_state = checkpoint["model_state_dict"]

    # Map BC encoder weights to PPO feature extractor
    # BC: encoder.0, encoder.2, encoder.4, encoder.6 (Conv2d layers)
    # PPO: policy.features_extractor.cnn.0, .2, .4, .6
    ppo_state = model.policy.state_dict()

    weights_loaded = 0
    for bc_key, bc_tensor in bc_state.items():
        if bc_key.startswith("encoder."):
            # Map encoder.X -> features_extractor.cnn.X
            ppo_key = bc_key.replace("encoder.", "features_extractor.cnn.")
            if ppo_key in ppo_state:
                if ppo_state[ppo_key].shape == bc_tensor.shape:
                    ppo_state[ppo_key] = bc_tensor
                    weights_loaded += 1
                    print(f"  ✓ {bc_key} -> {ppo_key}")
                else:
                    print(f"  ✗ {bc_key}: shape mismatch {bc_tensor.shape} vs {ppo_state[ppo_key].shape}")

    # Load the updated state dict
    model.policy.load_state_dict(ppo_state)
    print(f"Loaded {weights_loaded} weight tensors from BC checkpoint")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--eval-freq", type=int, default=50000)
    parser.add_argument("--output", default="models/vision_rl/best_model.zip")
    parser.add_argument("--bc-checkpoint", default="data/dagger/iter_02/model/best_model.pt",
                        help="BC checkpoint to initialize from")
    args = parser.parse_args()

    print("=" * 60)
    print("RL FINE-TUNING FOR VISION-BASED DRONE RACING")
    print("=" * 60)
    print(f"Timesteps: {args.timesteps}")
    print(f"Parallel envs: {args.envs}")
    print(f"Eval frequency: {args.eval_freq}")
    print()

    # Create parallel envs - THIS IS KEY FOR HARDWARE EFFICIENCY
    print(f"Creating {args.envs} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(args.envs)])
    env = VecMonitor(env)
    print(f"  ✓ {args.envs} envs ready")

    # Create model with custom CNN
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,  # Larger batch for GPU efficiency
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cuda",
    )

    print(f"\nModel created on device: {model.device}")
    print(f"Policy: {model.policy.__class__.__name__}")

    # Load BC weights if checkpoint exists
    if args.bc_checkpoint and Path(args.bc_checkpoint).exists():
        model = load_bc_weights_into_ppo(model, args.bc_checkpoint)
    else:
        print(f"\nWARNING: BC checkpoint not found at {args.bc_checkpoint}")
        print("Training from scratch (this is much slower!)")

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Initial evaluation
    print("\n--- Initial Evaluation (before training) ---")
    gates = evaluate_gates(model, n_episodes=5)
    print(f"Gates passed: {gates}, avg: {np.mean(gates):.1f}/5")

    # Train
    print(f"\n--- Training for {args.timesteps} steps ---")
    callback = ProgressCallback(eval_freq=args.eval_freq)

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=False,
    )

    # Save model
    model.save(args.output)
    print(f"\nModel saved to {args.output}")

    # Final evaluation
    print("\n--- Final Evaluation ---")
    gates = evaluate_gates(model, n_episodes=20)
    print(f"Gates passed: {gates}")
    print(f"Average: {np.mean(gates):.1f}/5")
    print(f"Best: {max(gates)}/5")
    print(f"Success rate (5/5): {100 * gates.count(5) / len(gates):.0f}%")

    env.close()
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
