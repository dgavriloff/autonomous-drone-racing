#!/usr/bin/env python3
"""
Train GateNet for gate segmentation.

Usage:
    python scripts/train_gate_net.py --data_dir data/collected --output_dir models/gate_net
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.vision.data_collector import DataCollector, GateDataset, collect_from_simulation
from src.vision.gate_net import GateNet, GateNetTrainer, GateNetDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train GateNet for gate segmentation")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/collected",
                        help="Directory containing collected data")
    parser.add_argument("--collect_data", action="store_true",
                        help="Collect new data before training")
    parser.add_argument("--num_frames", type=int, default=50000,
                        help="Number of frames to collect")

    # Model
    parser.add_argument("--encoder_channels", type=int, nargs="+", default=[32, 64, 128],
                        help="Encoder channel sizes")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")

    # Output
    parser.add_argument("--output_dir", type=str, default="models/gate_net",
                        help="Directory to save model")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda, mps)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("GateNet Training")
    print("=" * 60)

    # Collect data if requested
    if args.collect_data:
        print(f"\nCollecting {args.num_frames} frames of training data...")
        Path(args.data_dir).mkdir(parents=True, exist_ok=True)
        collect_from_simulation(
            output_dir=args.data_dir,
            num_frames=args.num_frames,
        )

    # Load data
    print(f"\nLoading data from {args.data_dir}...")

    try:
        dataset = GateDataset(args.data_dir, augment=True)
    except FileNotFoundError:
        print(f"Error: Data not found in {args.data_dir}")
        print("Run with --collect_data to collect training data first.")
        return 1

    print(f"Loaded {len(dataset)} samples")

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create model
    print("\nCreating model...")
    model = GateNet(
        encoder_channels=args.encoder_channels,
        dropout=args.dropout,
    )
    print(f"Parameters: {model.count_parameters():,}")

    # Create trainer
    trainer = GateNetTrainer(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(f"Device: {trainer.device}")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.output_dir,
        verbose=True,
    )

    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final train IoU: {history['train_iou'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"Final val IoU: {history['val_iou'][-1]:.4f}")
    print(f"\nModel saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
