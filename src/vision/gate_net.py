"""
GateNet - Lightweight U-Net for gate segmentation.

A compact U-Net architecture (<500K parameters) designed for real-time
gate detection on 64x48 images from drone cameras.

Architecture:
- Encoder: 3 down blocks with configurable channels
- Bottleneck: Feature compression
- Decoder: 3 up blocks with skip connections
- Output: Single channel probability map

Designed for:
- Input: RGB images (64x48 or configurable)
- Output: Binary segmentation mask (gate vs background)
- Real-time inference on embedded hardware
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    - Encoder: Configurable down blocks (default: 16 -> 32 -> 64 channels)
    - Bottleneck: Feature compression (default: 128 channels)
    - Decoder: Symmetric up blocks with skip connections
    - Output: Single channel probability map

    Total parameters: ~483K with default config (under 500K target for real-time inference)

    Default configuration:
    - Input: (B, 3, 48, 64) RGB images
    - Output: (B, 1, 48, 64) gate segmentation mask
    - encoder_channels: [16, 32, 64]
    - bottleneck_channels: 128

    The network is fully convolutional and can handle any input size
    that is divisible by 2^(num_encoder_blocks + 1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_channels: List[int] = None,
        bottleneck_channels: int = 128,
        dropout: float = 0.1,
        use_sigmoid: bool = True,
    ):
        """
        Initialize GateNet.

        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary segmentation)
            encoder_channels: Channel sizes for encoder blocks.
                              Default: [16, 32, 64] (~483K params, under 500K target)
            bottleneck_channels: Channels in bottleneck layer (default: 128)
            dropout: Dropout rate for regularization
            use_sigmoid: Whether to apply sigmoid to output (set False for BCEWithLogitsLoss)
        """
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [16, 32, 64]  # ~483K params with bottleneck=128

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.bottleneck_channels = bottleneck_channels
        self.use_sigmoid = use_sigmoid

        # Initial convolution
        self.initial = ConvBlock(in_channels, encoder_channels[0], dropout)

        # Encoder path
        self.encoders = nn.ModuleList()
        in_ch = encoder_channels[0]
        for out_ch in encoder_channels[1:]:
            self.encoders.append(DownBlock(in_ch, out_ch, dropout))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = DownBlock(encoder_channels[-1], bottleneck_channels, dropout)

        # Decoder path
        decoder_channels = encoder_channels[::-1]  # Reverse order
        self.decoders = nn.ModuleList()

        in_ch = bottleneck_channels
        for out_ch in decoder_channels:
            self.decoders.append(UpBlock(in_ch, out_ch, dropout))
            in_ch = out_ch

        # Final output layer
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)
        self.final_activation = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Gate segmentation mask (B, out_channels, H, W)
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
        x = self.final_conv(x)
        return self.final_activation(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving/loading."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "encoder_channels": self.encoder_channels,
            "bottleneck_channels": self.bottleneck_channels,
            "use_sigmoid": self.use_sigmoid,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GateNet":
        """Create model from configuration dict."""
        return cls(**config)


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.

    Measures overlap between prediction and ground truth.
    Good for imbalanced datasets where background dominates.
    """

    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
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


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    Focuses learning on hard examples by down-weighting easy ones.
    Useful when gates are small relative to background.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)

        # Compute focal weight
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Binary cross entropy
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss for segmentation.

    BCE provides pixel-wise accuracy while Dice handles class imbalance.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            bce_weight: Weight for BCE/Focal loss
            dice_weight: Weight for Dice loss
            use_focal: Use Focal loss instead of BCE
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if use_focal:
            self.bce = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
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


class SegmentationMetrics:
    """Compute segmentation metrics for evaluation."""

    @staticmethod
    def compute_iou(
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute Intersection over Union (Jaccard Index).

        Args:
            pred: Predicted mask (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
            threshold: Binarization threshold

        Returns:
            IoU score [0, 1]
        """
        pred_binary = (pred > threshold).float()

        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection

        if union == 0:
            return 1.0  # Both empty
        return (intersection / union).item()

    @staticmethod
    def compute_precision_recall(
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Compute precision and recall.

        Args:
            pred: Predicted mask (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
            threshold: Binarization threshold

        Returns:
            Tuple of (precision, recall)
        """
        pred_binary = (pred > threshold).float()

        true_positive = (pred_binary * target).sum()
        predicted_positive = pred_binary.sum()
        actual_positive = target.sum()

        precision = (true_positive / predicted_positive).item() if predicted_positive > 0 else 0.0
        recall = (true_positive / actual_positive).item() if actual_positive > 0 else 0.0

        return precision, recall

    @staticmethod
    def compute_dice(
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute Dice coefficient (F1 score).

        Args:
            pred: Predicted mask (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
            threshold: Binarization threshold

        Returns:
            Dice score [0, 1]
        """
        pred_binary = (pred > threshold).float()

        intersection = (pred_binary * target).sum()
        total = pred_binary.sum() + target.sum()

        if total == 0:
            return 1.0
        return (2 * intersection / total).item()

    @staticmethod
    def compute_all(
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute all metrics at once.

        Returns:
            Dict with 'iou', 'dice', 'precision', 'recall'
        """
        iou = SegmentationMetrics.compute_iou(pred, target, threshold)
        dice = SegmentationMetrics.compute_dice(pred, target, threshold)
        precision, recall = SegmentationMetrics.compute_precision_recall(pred, target, threshold)

        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }


class GateNetTrainer:
    """Training manager for GateNet with comprehensive metrics."""

    def __init__(
        self,
        model: GateNet,
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_focal_loss: bool = False,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        """
        Initialize trainer.

        Args:
            model: GateNet model to train
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            use_focal_loss: Use focal loss for better handling of class imbalance
            bce_weight: Weight for BCE component of loss
            dice_weight: Weight for Dice component of loss
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
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = CombinedLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            use_focal=use_focal_loss,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        # Extended history with more metrics
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_iou": [],
            "train_dice": [],
            "train_precision": [],
            "train_recall": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "val_precision": [],
            "val_recall": [],
            "learning_rate": [],
        }

    def train_epoch(
        self,
        train_loader,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dict with loss, iou, dice, precision, recall
        """
        self.model.train()
        total_loss = 0.0
        total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
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

            # Compute metrics
            with torch.no_grad():
                metrics = SegmentationMetrics.compute_all(pred, mask)
                for key in total_metrics:
                    total_metrics[key] += metrics[key]

            num_batches += 1

        # Average metrics
        result = {"loss": total_loss / num_batches}
        for key in total_metrics:
            result[key] = total_metrics[key] / num_batches

        return result

    def validate(
        self,
        val_loader,
    ) -> Dict[str, float]:
        """
        Validate model.

        Returns:
            Dict with loss, iou, dice, precision, recall
        """
        self.model.eval()
        total_loss = 0.0
        total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
        num_batches = 0

        with torch.no_grad():
            for rgb, mask in val_loader:
                rgb = rgb.to(self.device)
                mask = mask.to(self.device)

                pred = self.model(rgb)
                loss = self.criterion(pred, mask)

                total_loss += loss.item()

                # Compute metrics
                metrics = SegmentationMetrics.compute_all(pred, mask)
                for key in total_metrics:
                    total_metrics[key] += metrics[key]

                num_batches += 1

        # Average metrics
        result = {"loss": total_loss / num_batches}
        for key in total_metrics:
            result[key] = total_metrics[key] / num_batches

        return result

    def train(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 10,
        save_dir: Optional[str] = None,
        verbose: bool = True,
        early_stopping_patience: int = 0,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            verbose: Whether to print progress
            early_stopping_patience: Stop if val_loss doesn't improve for N epochs (0=disabled)

        Returns:
            Training history
        """
        best_val_iou = 0.0
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_iou"].append(train_metrics["iou"])
            self.history["train_dice"].append(train_metrics["dice"])
            self.history["train_precision"].append(train_metrics["precision"])
            self.history["train_recall"].append(train_metrics["recall"])

            # Record learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            # Validate
            val_metrics = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_iou"].append(val_metrics["iou"])
                self.history["val_dice"].append(val_metrics["dice"])
                self.history["val_precision"].append(val_metrics["precision"])
                self.history["val_recall"].append(val_metrics["recall"])
                self.scheduler.step(val_metrics["loss"])

            # Print progress
            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - "
                msg += f"Train: loss={train_metrics['loss']:.4f}, IoU={train_metrics['iou']:.4f}"
                if val_loader is not None:
                    msg += f" | Val: loss={val_metrics['loss']:.4f}, IoU={val_metrics['iou']:.4f}"
                msg += f" | lr={current_lr:.2e}"
                print(msg)

            # Save best model
            if save_dir is not None and val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                self.save(save_dir, "best_model.pt")

            # Early stopping
            if early_stopping_patience > 0 and val_loader is not None:
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping triggered after {epoch + 1} epochs")
                        break

        # Save final model
        if save_dir is not None:
            self.save(save_dir, "final_model.pt")

        return self.history

    def save(self, save_dir: str, filename: str = "model.pt"):
        """Save model checkpoint."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "config": self.model.get_config(),
        }

        torch.save(checkpoint, save_dir / filename)
        if filename == "best_model.pt":
            print(f"  Saved best model to {save_dir / filename}")

    def load(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
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
    from torch.utils.data import DataLoader, TensorDataset

    # Test GateNet architecture
    print("Testing GateNet architecture...")
    print("=" * 50)

    model = GateNet()
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Model config: {model.get_config()}")

    # Test forward pass
    batch = torch.randn(2, 3, 48, 64)
    output = model(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Test with random data
    print("\n" + "=" * 50)
    print("Testing training pipeline...")

    # Create random tensor data
    num_samples = 100
    images = torch.rand(num_samples, 3, 48, 64)
    masks = (torch.rand(num_samples, 1, 48, 64) > 0.8).float()

    dataset = TensorDataset(images, masks)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    trainer = GateNetTrainer(model, device="cpu")
    history = trainer.train(loader, epochs=2, verbose=True)

    print("\n" + "=" * 50)
    print("Test complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final train IoU: {history['train_iou'][-1]:.4f}")
    print(f"Final train precision: {history['train_precision'][-1]:.4f}")
    print(f"Final train recall: {history['train_recall'][-1]:.4f}")
