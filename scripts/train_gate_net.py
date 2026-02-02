#!/usr/bin/env python3
"""
Train GateNet for gate segmentation.

This script provides a complete training pipeline for the GateNet
vision model used for drone racing gate detection.

Usage:
    # Train with collected data
    python scripts/train_gate_net.py --data_dir data/collected --epochs 50

    # Collect new data and train
    python scripts/train_gate_net.py --collect_data --num_frames 50000 --epochs 50

    # Train with synthetic data (for testing)
    python scripts/train_gate_net.py --synthetic --num_samples 5000 --epochs 20

    # Use focal loss for better handling of class imbalance
    python scripts/train_gate_net.py --data_dir data/collected --use_focal

Example on training PC:
    python scripts/train_gate_net.py --data_dir data/collected \\
        --epochs 100 --batch_size 64 --device cuda
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.vision.gate_net import GateNet, GateNetTrainer, SegmentationMetrics
from src.vision.data_loader import (
    GateSegmentationDataset,
    create_dataloaders,
    create_synthetic_data,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GateNet for gate segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_dir", type=str, default="data/collected",
        help="Directory containing collected data"
    )
    data_group.add_argument(
        "--collect_data", action="store_true",
        help="Collect new data before training"
    )
    data_group.add_argument(
        "--num_frames", type=int, default=50000,
        help="Number of frames to collect (with --collect_data)"
    )
    data_group.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for testing"
    )
    data_group.add_argument(
        "--num_samples", type=int, default=5000,
        help="Number of synthetic samples (with --synthetic)"
    )

    # Model architecture
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--encoder_channels", type=int, nargs="+", default=[16, 32, 64],
        help="Encoder channel sizes (default gives ~483K params)"
    )
    model_group.add_argument(
        "--bottleneck_channels", type=int, default=128,
        help="Bottleneck channel size"
    )
    model_group.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate"
    )

    # Training hyperparameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size"
    )
    train_group.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Initial learning rate"
    )
    train_group.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay (L2 regularization)"
    )
    train_group.add_argument(
        "--val_split", type=float, default=0.1,
        help="Validation split ratio"
    )
    train_group.add_argument(
        "--early_stopping", type=int, default=10,
        help="Early stopping patience (0 to disable)"
    )

    # Loss configuration
    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument(
        "--use_focal", action="store_true",
        help="Use focal loss instead of BCE"
    )
    loss_group.add_argument(
        "--bce_weight", type=float, default=0.5,
        help="Weight for BCE/Focal loss component"
    )
    loss_group.add_argument(
        "--dice_weight", type=float, default=0.5,
        help="Weight for Dice loss component"
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir", type=str, default="models/gate_net",
        help="Directory to save model and logs"
    )
    output_group.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (auto, cpu, cuda, mps)"
    )
    output_group.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of data loading workers"
    )
    output_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_training_config(args, output_dir: Path, model: GateNet):
    """Save training configuration for reproducibility."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "model_config": model.get_config(),
        "model_parameters": model.count_parameters(),
    }

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)


def plot_training_history(history: dict, output_dir: Path):
    """Plot and save training curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train")
    if history["val_loss"]:
        ax.plot(history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True)

    # IoU
    ax = axes[0, 1]
    ax.plot(history["train_iou"], label="Train")
    if history["val_iou"]:
        ax.plot(history["val_iou"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Intersection over Union")
    ax.legend()
    ax.grid(True)

    # Precision/Recall
    ax = axes[1, 0]
    ax.plot(history["train_precision"], label="Train Precision")
    ax.plot(history["train_recall"], label="Train Recall")
    if history["val_precision"]:
        ax.plot(history["val_precision"], label="Val Precision", linestyle="--")
        ax.plot(history["val_recall"], label="Val Recall", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall")
    ax.legend()
    ax.grid(True)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(history["learning_rate"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()

    print(f"Training curves saved to {output_dir / 'training_curves.png'}")


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("GateNet Training")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    if args.synthetic:
        print(f"\nGenerating {args.num_samples} synthetic samples...")
        images, masks = create_synthetic_data(
            num_samples=args.num_samples,
            image_size=(64, 48),
        )
        data_source = (images, masks)
        print(f"Generated {len(images)} samples")

    elif args.collect_data:
        print(f"\nCollecting {args.num_frames} frames of training data...")
        from src.vision.data_collector import collect_from_simulation

        Path(args.data_dir).mkdir(parents=True, exist_ok=True)
        collect_from_simulation(
            output_dir=args.data_dir,
            num_frames=args.num_frames,
        )
        data_source = args.data_dir

    else:
        data_source = args.data_dir

    # Create data loaders
    print(f"\nLoading data from {data_source if isinstance(data_source, str) else 'memory'}...")

    try:
        train_loader, val_loader = create_dataloaders(
            data_source=data_source,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            augment_train=True,
            augment_val=False,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run with --collect_data to collect training data first,")
        print("or use --synthetic for testing with synthetic data.")
        return 1

    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = GateNet(
        encoder_channels=args.encoder_channels,
        bottleneck_channels=args.bottleneck_channels,
        dropout=args.dropout,
    )
    print(f"Parameters: {model.count_parameters():,}")

    # Save training config
    save_training_config(args, output_dir, model)

    # Create trainer
    trainer = GateNetTrainer(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_focal_loss=args.use_focal,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
    )
    print(f"Device: {trainer.device}")
    print(f"Loss: {'Focal' if args.use_focal else 'BCE'} + Dice "
          f"(weights: {args.bce_weight:.2f}, {args.dice_weight:.2f})")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.output_dir,
        verbose=True,
        early_stopping_patience=args.early_stopping,
    )

    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    print("\nFinal Metrics:")
    print(f"  Train - Loss: {history['train_loss'][-1]:.4f}, "
          f"IoU: {history['train_iou'][-1]:.4f}, "
          f"Precision: {history['train_precision'][-1]:.4f}, "
          f"Recall: {history['train_recall'][-1]:.4f}")

    if history["val_loss"]:
        print(f"  Val   - Loss: {history['val_loss'][-1]:.4f}, "
              f"IoU: {history['val_iou'][-1]:.4f}, "
              f"Precision: {history['val_precision'][-1]:.4f}, "
              f"Recall: {history['val_recall'][-1]:.4f}")

        # Find best epoch
        best_epoch = np.argmax(history["val_iou"])
        print(f"\nBest validation IoU: {history['val_iou'][best_epoch]:.4f} (epoch {best_epoch + 1})")

    print(f"\nModel saved to: {output_dir}")

    # Plot training history
    plot_training_history(history, output_dir)

    # Save history as JSON
    history_json = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(output_dir / "history.json", "w") as f:
        json.dump(history_json, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
