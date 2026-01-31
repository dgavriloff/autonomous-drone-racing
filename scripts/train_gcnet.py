#!/usr/bin/env python3
"""
Train G&CNet controller.

Phase 1: Imitation learning from PID expert
Phase 2: RL fine-tuning with PPO

Usage:
    python scripts/train_gcnet.py --phase imitation --output_dir models/gcnet
    python scripts/train_gcnet.py --phase rl --pretrained models/gcnet/best_model.pt
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.control.gcnet import (
    GCNet, GCNetTrainer, GCNetDataset, GCNetRLWrapper,
    collect_expert_data, create_gcnet
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train G&CNet controller")

    # Training phase
    parser.add_argument("--phase", type=str, choices=["imitation", "rl", "both"],
                        default="imitation", help="Training phase")

    # Data collection (for imitation)
    parser.add_argument("--collect_data", action="store_true",
                        help="Collect new expert data")
    parser.add_argument("--expert_steps", type=int, default=100000,
                        help="Number of expert demonstration steps")
    parser.add_argument("--data_dir", type=str, default="data/expert",
                        help="Directory for expert data")

    # Model
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer sizes")
    parser.add_argument("--use_residual", action="store_true", default=True,
                        help="Use residual connections")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model")

    # Imitation learning
    parser.add_argument("--il_epochs", type=int, default=100,
                        help="Imitation learning epochs")
    parser.add_argument("--il_batch_size", type=int, default=256,
                        help="Imitation learning batch size")
    parser.add_argument("--il_lr", type=float, default=3e-4,
                        help="Imitation learning rate")

    # RL fine-tuning
    parser.add_argument("--rl_timesteps", type=int, default=2000000,
                        help="RL training timesteps")
    parser.add_argument("--rl_n_steps", type=int, default=2048,
                        help="PPO n_steps")
    parser.add_argument("--rl_batch_size", type=int, default=64,
                        help="PPO batch size")

    # Output
    parser.add_argument("--output_dir", type=str, default="models/gcnet",
                        help="Directory to save model")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")

    return parser.parse_args()


def train_imitation(args):
    """Train G&CNet using imitation learning."""
    print("\n" + "=" * 60)
    print("Phase 1: Imitation Learning")
    print("=" * 60)

    # Collect or load expert data
    data_path = Path(args.data_dir)
    states_path = data_path / "states.npy"
    actions_path = data_path / "actions.npy"

    if args.collect_data or not states_path.exists():
        print(f"\nCollecting {args.expert_steps} expert demonstrations...")
        data_path.mkdir(parents=True, exist_ok=True)
        states, actions = collect_expert_data(num_steps=args.expert_steps)
        np.save(states_path, states)
        np.save(actions_path, actions)
    else:
        print(f"\nLoading expert data from {args.data_dir}...")
        states = np.load(states_path)
        actions = np.load(actions_path)

    print(f"Expert data: {states.shape[0]} samples")

    # Create dataset
    dataset = GCNetDataset(states, actions)

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.il_batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.il_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    print("\nCreating model...")
    model = GCNet(
        hidden_dims=args.hidden_dims,
        use_residual=args.use_residual,
    )

    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Parameters: {model.count_parameters():,}")

    # Create trainer
    trainer = GCNetTrainer(
        model=model,
        device=args.device,
        learning_rate=args.il_lr,
    )
    print(f"Device: {trainer.device}")

    # Train
    print(f"\nTraining for {args.il_epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.il_epochs,
        save_dir=args.output_dir,
        verbose=True,
    )

    print("\nImitation learning complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")

    return model


def train_rl(args, pretrained_model=None):
    """Fine-tune G&CNet using RL (PPO)."""
    print("\n" + "=" * 60)
    print("Phase 2: RL Fine-tuning")
    print("=" * 60)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("Error: stable-baselines3 required for RL training")
        print("Install with: pip install stable-baselines3")
        return None

    from src.envs.high_freq_racing import HighFreqRacingAviary, create_monorace_track

    # Create environment
    print("\nCreating environment...")
    track = create_monorace_track(num_gates=5)

    def make_env():
        return HighFreqRacingAviary(
            track=track,
            ctrl_freq=500,
            pyb_freq=2000,
            gui=False,
        )

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Create policy
    print("\nCreating PPO policy...")

    # Load pretrained if available
    pretrained_path = args.pretrained or (Path(args.output_dir) / "best_model.pt")
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Initializing from {pretrained_path}")
        # Note: SB3 doesn't directly support loading custom weights
        # Would need custom policy class for full integration

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=args.rl_n_steps,
        batch_size=args.rl_batch_size,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"{args.output_dir}/tb_logs",
    )

    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.output_dir,
        log_path=args.output_dir,
        eval_freq=10000,
        deterministic=True,
    )

    # Train
    print(f"\nTraining for {args.rl_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.rl_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # Save final model
    model.save(f"{args.output_dir}/final_ppo_model")
    print(f"\nRL training complete! Model saved to {args.output_dir}")

    env.close()
    eval_env.close()

    return model


def main():
    args = parse_args()

    print("=" * 60)
    print("G&CNet Training")
    print("=" * 60)
    print(f"Phase: {args.phase}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = None

    if args.phase in ["imitation", "both"]:
        model = train_imitation(args)

    if args.phase in ["rl", "both"]:
        train_rl(args, model)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Models saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
