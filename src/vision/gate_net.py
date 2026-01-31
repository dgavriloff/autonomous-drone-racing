"""
GateNet - Lightweight U-Net for gate segmentation.

A compact U-Net architecture (<500K parameters) designed for real-time
gate detection on 64x48 images from drone cameras.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ConvBlock(nn.Module):
    """Double convolution block with batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Encoder block: MaxPool + ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Decoder block: Upsample + Concat + ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle size mismatch from odd dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class GateNet(nn.Module):
    """
    Lightweight U-Net for gate segmentation.

    Architecture:
    - Encoder: 3 down blocks (32 -> 64 -> 128 channels)
    - Bottleneck: 256 channels
    - Decoder: 3 up blocks (128 -> 64 -> 32 channels)
    - Output: 1 channel (gate probability)

    Total parameters: ~400K (under 500K target)

    Input: (B, 3, 48, 64) RGB images
    Output: (B, 1, 48, 64) gate segmentation mask
    """

    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: List[int] = [32, 64, 128],
        bottleneck_channels: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize GateNet.

        Args:
            in_channels: Number of input channels (3 for RGB)
            encoder_channels: Channel sizes for encoder blocks
            bottleneck_channels: Channels in bottleneck
            dropout: Dropout rate
        """
        super().__init__()

        self.encoder_channels = encoder_channels
        self.bottleneck_channels = bottleneck_channels

        # Initial convolution
        self.initial = ConvBlock(in_channels, encoder_channels[0], dropout)

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = encoder_channels[0]
        for out_ch in encoder_channels[1:]:
            self.encoders.append(DownBlock(in_ch, out_ch, dropout))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = DownBlock(encoder_channels[-1], bottleneck_channels, dropout)

        # Decoder
        decoder_channels = encoder_channels[::-1]  # Reverse order
        self.decoders = nn.ModuleList()

        in_ch = bottleneck_channels
        for out_ch in decoder_channels:
            self.decoders.append(UpBlock(in_ch, out_ch, dropout))
            in_ch = out_ch

        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Gate segmentation mask (B, 1, H, W)
        """
        # Encoder path with skip connections
        skips = []

        x = self.initial(x)
        skips.append(x)

        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skips = skips[::-1]  # Reverse for decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[i])

        # Final output
        return self.final(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class GateNetDataset(Dataset):
    """PyTorch Dataset wrapper for GateNet training."""

    def __init__(
        self,
        rgb_images: np.ndarray,
        masks: np.ndarray,
        augment: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            rgb_images: (N, H, W, 3) RGB images
            masks: (N, H, W) binary masks
            augment: Whether to apply augmentation
        """
        self.rgb_images = rgb_images
        self.masks = masks
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rgb_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb = self.rgb_images[idx].copy()
        mask = self.masks[idx].copy()

        # Apply augmentation
        if self.augment:
            rgb, mask = self._augment(rgb, mask)

        # Normalize RGB to [0, 1] and convert to (C, H, W)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))

        # Ensure mask is float and add channel dimension
        mask = mask.astype(np.float32)
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]

        return torch.from_numpy(rgb), torch.from_numpy(mask)

    def _augment(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            rgb = np.fliplr(rgb).copy()
            mask = np.fliplr(mask).copy()

        # Random brightness
        brightness = np.random.uniform(0.8, 1.2)
        rgb = np.clip(rgb * brightness, 0, 255).astype(np.uint8)

        # Random contrast
        contrast = np.random.uniform(0.8, 1.2)
        mean = rgb.mean()
        rgb = np.clip((rgb - mean) * contrast + mean, 0, 255).astype(np.uint8)

        # Random Gaussian noise
        if np.random.random() > 0.5:
            noise_std = np.random.uniform(0, 10)
            noise = np.random.normal(0, noise_std, rgb.shape)
            rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)

        return rgb, mask


class GateNetTrainer:
    """Training manager for GateNet."""

    def __init__(
        self,
        model: GateNet,
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        """
        Initialize trainer.

        Args:
            model: GateNet model to train
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
        """
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_iou": [],
            "val_iou": [],
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average loss, average IoU)
        """
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0

        for rgb, mask in train_loader:
            rgb = rgb.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(rgb)
            loss = self.criterion(pred, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_iou += self._compute_iou(pred, mask)
            num_batches += 1

        return total_loss / num_batches, total_iou / num_batches

    def validate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, float]:
        """
        Validate model.

        Returns:
            Tuple of (average loss, average IoU)
        """
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0

        with torch.no_grad():
            for rgb, mask in val_loader:
                rgb = rgb.to(self.device)
                mask = mask.to(self.device)

                pred = self.model(rgb)
                loss = self.criterion(pred, mask)

                total_loss += loss.item()
                total_iou += self._compute_iou(pred, mask)
                num_batches += 1

        return total_loss / num_batches, total_iou / num_batches

    def _compute_iou(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """Compute Intersection over Union."""
        pred_binary = (pred > threshold).float()

        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection

        if union == 0:
            return 1.0  # Both empty
        return (intersection / union).item()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            verbose: Whether to print progress

        Returns:
            Training history
        """
        best_val_iou = 0.0

        for epoch in range(epochs):
            # Train
            train_loss, train_iou = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_iou"].append(train_iou)

            # Validate
            val_loss, val_iou = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_iou = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_iou"].append(val_iou)
                self.scheduler.step(val_loss)

            # Print progress
            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - "
                msg += f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}"
                if val_loader is not None:
                    msg += f" | Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}"
                print(msg)

            # Save best model
            if save_dir is not None and val_iou > best_val_iou:
                best_val_iou = val_iou
                self.save(save_dir, f"best_model.pt")

        # Save final model
        if save_dir is not None:
            self.save(save_dir, f"final_model.pt")

        return self.history

    def save(self, save_dir: str, filename: str = "model.pt"):
        """Save model checkpoint."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": {
                "encoder_channels": self.model.encoder_channels,
                "bottleneck_channels": self.model.bottleneck_channels,
            },
        }

        torch.save(checkpoint, save_dir / filename)

    def load(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)


def create_gatenet(
    pretrained_path: Optional[str] = None,
    device: str = "auto",
) -> GateNet:
    """
    Create GateNet model, optionally loading pretrained weights.

    Args:
        pretrained_path: Path to pretrained checkpoint
        device: Device to use

    Returns:
        GateNet model
    """
    model = GateNet()

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    # Test GateNet architecture
    print("Testing GateNet architecture...")

    model = GateNet()
    print(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass
    batch = torch.randn(2, 3, 48, 64)
    output = model(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {output.shape}")

    # Test with random data
    print("\nTesting training pipeline...")

    # Create random data
    num_samples = 100
    rgb_images = np.random.randint(0, 255, (num_samples, 48, 64, 3), dtype=np.uint8)
    masks = (np.random.random((num_samples, 48, 64)) > 0.8).astype(np.float32)

    dataset = GateNetDataset(rgb_images, masks, augment=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    trainer = GateNetTrainer(model, device="cpu")
    history = trainer.train(loader, epochs=2, verbose=True)

    print("\nTest complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final train IoU: {history['train_iou'][-1]:.4f}")
