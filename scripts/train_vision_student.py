#!/usr/bin/env python3
"""
Train a vision-based student policy via behavioral cloning.

Takes teacher demonstrations (image, action pairs) and trains a student
network to predict actions from camera images.

Architecture:
    Camera Image (64x48x3)
        -> GateNet Encoder (frozen or fine-tuned)
        -> MLP Policy Head (256, 256)
        -> Action (4D velocity command)
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
import argparse
from typing import Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.gate_net import GateNet, create_gatenet


class VisionStudentNet(nn.Module):
    """
    Vision-based student policy network.

    Uses GateNet encoder (frozen or fine-tuned) + MLP policy head.
    """

    def __init__(
        self,
        gatenet_path: str = None,
        action_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (256, 256),
        freeze_encoder: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.freeze_encoder = freeze_encoder

        # Load GateNet as encoder
        if gatenet_path:
            self.encoder = create_gatenet(gatenet_path, device=device)
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
        else:
            # Create new encoder (same architecture as GateNet)
            self.encoder = GateNet(in_channels=3, base_channels=32)

        # Determine encoder output size by forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 48, 64).to(device)
            encoder_out = self._get_encoder_features(dummy)
            encoder_dim = encoder_out.shape[1]

        # Policy head MLP
        layers = []
        in_dim = encoder_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())  # Actions in [-1, 1]

        self.policy_head = nn.Sequential(*layers)

        print(f"VisionStudentNet: encoder_dim={encoder_dim}, action_dim={action_dim}")
        print(f"  Encoder frozen: {freeze_encoder}")
        print(f"  Total params: {sum(p.numel() for p in self.parameters())}")
        print(f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def _get_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract flattened features from encoder."""
        # Get the segmentation mask output
        with torch.set_grad_enabled(not self.freeze_encoder):
            mask = self.encoder(x)  # (B, 1, H, W)

        # Also get intermediate features by hooking into encoder
        # For now, just flatten the mask output
        features = mask.view(mask.size(0), -1)  # (B, H*W)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W), normalized to [0, 1]

        Returns:
            Actions (B, action_dim) in [-1, 1]
        """
        features = self._get_encoder_features(x)
        actions = self.policy_head(features)
        return actions

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Predict actions from images (for inference)."""
        self.eval()
        with torch.no_grad():
            actions = self.forward(x)
        return actions.cpu().numpy()


class VisionStudentNetV2(nn.Module):
    """
    Vision-based student policy with deeper encoder features.

    Uses intermediate GateNet features instead of just output mask.
    """

    def __init__(
        self,
        gatenet_path: str = None,
        action_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (256, 256),
        freeze_encoder: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.freeze_encoder = freeze_encoder

        # Build encoder from scratch (simpler CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 48x64 -> 24x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 24x32 -> 12x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 12x16 -> 6x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 6x8 -> 3x4
            nn.ReLU(),
            nn.Flatten(),  # 256 * 3 * 4 = 3072
        )

        # Note: GateNet uses a different architecture (U-Net with DownBlocks)
        # so we train the encoder from scratch instead of transferring weights
        if gatenet_path and Path(gatenet_path).exists():
            print(f"GateNet path provided but using fresh encoder (architecture mismatch)")

        # Compute encoder output dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 48, 64)
            encoder_dim = self.encoder(dummy).shape[1]

        # Policy head
        layers = []
        in_dim = encoder_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())

        self.policy_head = nn.Sequential(*layers)

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VisionStudentNetV2: encoder_dim={encoder_dim}")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        actions = self.policy_head(features)
        return actions

    def predict(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            actions = self.forward(x)
        return actions.cpu().numpy()


class DemoDataset(Dataset):
    """Dataset for behavioral cloning from teacher demonstrations."""

    def __init__(self, demos_path: str, augment: bool = True):
        self.demos_path = Path(demos_path)
        self.augment = augment

        # Load demonstrations
        with open(self.demos_path / "demos.json", "r") as f:
            self.demos = json.load(f)

        print(f"Loaded {len(self.demos)} demonstration frames")

    def __len__(self) -> int:
        return len(self.demos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        demo = self.demos[idx]

        # Load image
        rgb = np.load(demo["image_path"])
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]

        # Apply augmentation
        if self.augment:
            rgb = self._augment(rgb)

        # Normalize to [0, 1] and convert to tensor
        rgb = rgb.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)

        # Get action (handle nested list from SB3 predict)
        action = np.array(demo["action"]).squeeze()
        action = torch.tensor(action, dtype=torch.float32)

        return rgb_tensor, action

    def _augment(self, rgb: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Random brightness
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            rgb = np.clip(rgb * factor, 0, 255).astype(np.uint8)

        # Random noise
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 5, rgb.shape)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random horizontal flip (with action flip - but velocity doesn't flip simply)
        # Skip flip for now to avoid action space issues

        return rgb


def train_student(
    demos_path: str,
    output_path: str,
    gatenet_path: str = None,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    freeze_encoder: bool = True,
    val_split: float = 0.1,
    device: str = None,
):
    """
    Train vision student via behavioral cloning.

    Args:
        demos_path: Path to teacher demonstrations
        output_path: Path to save trained model
        gatenet_path: Optional path to pretrained GateNet
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: L2 regularization
        freeze_encoder: Whether to freeze encoder
        val_split: Validation split ratio
        device: Device to use
    """
    print("=" * 70)
    print("TRAINING VISION STUDENT")
    print("=" * 70)

    # Setup device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}")

    # Load dataset
    dataset = DemoDataset(demos_path, augment=True)

    # Train/val split
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")

    # Create model
    model = VisionStudentNetV2(
        gatenet_path=gatenet_path,
        action_dim=4,
        hidden_dims=(256, 256),
        freeze_encoder=freeze_encoder,
        device=device,
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for images, actions in train_loader:
            images = images.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            pred_actions = model(images)
            loss = criterion(pred_actions, actions)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, actions in val_loader:
                images = images.to(device)
                actions = actions.to(device)

                pred_actions = model(images)
                loss = criterion(pred_actions, actions)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, output_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, best={best_val_loss:.6f}")

    # Save final model and training history
    final_path = Path(output_path).with_suffix(".final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epochs,
        "history": history,
    }, final_path)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {output_path}")
    print(f"Final model saved to: {final_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train vision student policy")
    parser.add_argument("--demos", default="data/teacher_demos",
                        help="Path to teacher demonstrations")
    parser.add_argument("--output", default="models/vision_student/best_model.pt",
                        help="Output path for trained model")
    parser.add_argument("--gatenet", default="models/gate_net/best_model.pt",
                        help="Path to pretrained GateNet")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Fine-tune encoder (don't freeze)")
    parser.add_argument("--device", default=None,
                        help="Device (cuda, mps, cpu)")
    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    train_student(
        demos_path=args.demos,
        output_path=args.output,
        gatenet_path=args.gatenet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_encoder=not args.fine_tune,
        device=args.device,
    )


if __name__ == "__main__":
    main()
