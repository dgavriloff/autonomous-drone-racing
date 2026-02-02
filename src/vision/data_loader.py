"""
Data loading and augmentation for GateNet training.

Provides a flexible dataset class for loading image + mask pairs with
extensive data augmentation for robust gate segmentation training.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable, Union

import torch
from torch.utils.data import Dataset, DataLoader


class GateSegmentationDataset(Dataset):
    """
    PyTorch Dataset for gate segmentation training.

    Supports loading from:
    - Directory with images/ and masks/ subdirectories
    - NumPy arrays directly
    - DataCollector output format

    Implements extensive data augmentation for sim-to-real transfer.
    """

    def __init__(
        self,
        data_source: Union[str, Tuple[np.ndarray, np.ndarray]],
        image_size: Tuple[int, int] = (64, 48),
        augment: bool = True,
        augment_config: Optional[Dict] = None,
        normalize: bool = True,
        cache_in_memory: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_source: Either a path to data directory or tuple of (images, masks)
            image_size: Expected (width, height) of images
            augment: Whether to apply data augmentation
            augment_config: Optional dict to configure augmentation parameters
            normalize: Whether to normalize images to [0, 1]
            cache_in_memory: Whether to cache all data in memory
        """
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory

        # Default augmentation configuration
        self.augment_config = {
            # Photometric augmentations
            "brightness_range": (0.7, 1.3),
            "contrast_range": (0.7, 1.3),
            "saturation_range": (0.8, 1.2),
            "noise_std_range": (0, 15),
            "noise_prob": 0.5,
            # Geometric augmentations
            "horizontal_flip_prob": 0.5,
            "rotation_range": (-15, 15),  # degrees
            "rotation_prob": 0.3,
            "scale_range": (0.9, 1.1),
            "scale_prob": 0.3,
            # Blur augmentation
            "blur_prob": 0.2,
            "blur_kernel_sizes": [3, 5],
        }

        if augment_config:
            self.augment_config.update(augment_config)

        # Load data based on source type
        if isinstance(data_source, tuple):
            # Direct numpy arrays
            self.images, self.masks = data_source
            self.data_mode = "memory"
        elif isinstance(data_source, str):
            self._load_from_directory(data_source)
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

        # Validate data
        self._validate_data()

        # Cache for memory mode
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        if self.cache_in_memory and self.data_mode == "disk":
            self._preload_cache()

    def _load_from_directory(self, data_dir: str):
        """Load dataset from directory structure."""
        data_path = Path(data_dir)

        # Check for DataCollector format first
        if (data_path / "frames.json").exists():
            self._load_datacollector_format(data_path)
            return

        # Check for simple images/masks format
        images_dir = data_path / "images"
        masks_dir = data_path / "masks"

        if images_dir.exists() and masks_dir.exists():
            self._load_simple_format(images_dir, masks_dir)
            return

        # Check for flat directory with rgb_*.npy and mask_*.npy files
        rgb_files = sorted(data_path.glob("rgb_*.npy"))
        if rgb_files:
            self._load_flat_format(data_path)
            return

        raise FileNotFoundError(
            f"Could not find valid data in {data_dir}. "
            "Expected: frames.json, images/masks dirs, or rgb_*.npy files"
        )

    def _load_datacollector_format(self, data_path: Path):
        """Load from DataCollector output format."""
        with open(data_path / "frames.json", "r") as f:
            frames_data = json.load(f)

        self.image_paths = []
        self.mask_paths = []

        for frame in frames_data:
            # Get RGB path
            rgb_path = frame.get("rgb_path")
            if rgb_path and Path(rgb_path).exists():
                # Get combined mask from all visible gates
                gate_masks = frame.get("gate_masks", {})
                if gate_masks:
                    # Use first gate mask as reference (we'll combine later)
                    self.image_paths.append(rgb_path)
                    self.mask_paths.append(gate_masks)

        self.data_mode = "datacollector"
        print(f"Loaded {len(self.image_paths)} frames from DataCollector format")

    def _load_simple_format(self, images_dir: Path, masks_dir: Path):
        """Load from simple images/masks directory structure."""
        # Supported image formats
        image_extensions = {".npy", ".png", ".jpg", ".jpeg"}

        self.image_paths = []
        self.mask_paths = []

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() in image_extensions:
                # Find corresponding mask
                mask_name = img_path.stem
                for ext in image_extensions:
                    mask_path = masks_dir / f"{mask_name}{ext}"
                    if mask_path.exists():
                        self.image_paths.append(str(img_path))
                        self.mask_paths.append(str(mask_path))
                        break

        self.data_mode = "disk"
        print(f"Loaded {len(self.image_paths)} image-mask pairs")

    def _load_flat_format(self, data_path: Path):
        """Load from flat directory with rgb_*.npy and corresponding masks."""
        self.image_paths = []
        self.mask_paths = []

        masks_dir = data_path / "masks"

        for rgb_path in sorted(data_path.glob("rgb_*.npy")):
            frame_id = rgb_path.stem.split("_")[-1]
            # Look for any gate mask for this frame
            mask_pattern = f"gate_*_frame_{frame_id}.npy"
            mask_files = list(masks_dir.glob(mask_pattern))

            if mask_files:
                self.image_paths.append(str(rgb_path))
                # Store all mask paths for this frame
                self.mask_paths.append([str(m) for m in mask_files])

        self.data_mode = "flat"
        print(f"Loaded {len(self.image_paths)} frames from flat format")

    def _validate_data(self):
        """Validate loaded data."""
        if self.data_mode == "memory":
            if len(self.images) != len(self.masks):
                raise ValueError(
                    f"Number of images ({len(self.images)}) != "
                    f"number of masks ({len(self.masks)})"
                )
            if len(self.images) == 0:
                raise ValueError("No data provided")
        else:
            if len(self.image_paths) == 0:
                raise ValueError("No valid image-mask pairs found")

    def _preload_cache(self):
        """Preload all data into memory."""
        print("Preloading data into memory...")
        for idx in range(len(self)):
            self._cache[idx] = self._load_sample(idx)
        print(f"Cached {len(self._cache)} samples")

    def __len__(self) -> int:
        if self.data_mode == "memory":
            return len(self.images)
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, mask_tensor)
            - image_tensor: (3, H, W) float tensor in [0, 1]
            - mask_tensor: (1, H, W) float tensor in [0, 1]
        """
        # Load from cache or disk
        if idx in self._cache:
            image, mask = self._cache[idx]
            image = image.copy()
            mask = mask.copy()
        else:
            image, mask = self._load_sample(idx)

        # Apply augmentation
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # Normalize if needed
        if self.normalize and image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Convert to tensors
        # Image: (H, W, 3) -> (3, H, W)
        if image.ndim == 3:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())
        else:
            image_tensor = torch.from_numpy(image.copy())

        # Mask: (H, W) -> (1, H, W)
        if mask.ndim == 2:
            mask_tensor = torch.from_numpy(mask[np.newaxis, :, :].copy())
        else:
            mask_tensor = torch.from_numpy(mask.copy())

        return image_tensor.float(), mask_tensor.float()

    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single sample from disk or memory."""
        if self.data_mode == "memory":
            image = self.images[idx].copy()
            mask = self.masks[idx].copy()
        else:
            image = self._load_image(self.image_paths[idx])
            mask = self._load_mask(self.mask_paths[idx])

        return image, mask

    def _load_image(self, path: str) -> np.ndarray:
        """Load an image from file."""
        path = Path(path)

        if path.suffix == ".npy":
            image = np.load(path)
            # Remove alpha channel if present
            if image.shape[-1] == 4:
                image = image[..., :3]
        else:
            import cv2
            image = cv2.imread(str(path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _load_mask(self, path_data: Union[str, List[str], Dict]) -> np.ndarray:
        """Load and combine masks from file(s)."""
        # Handle different mask path formats
        if isinstance(path_data, dict):
            # DataCollector format: dict of gate_id -> path
            mask_paths = list(path_data.values())
        elif isinstance(path_data, list):
            # Flat format: list of paths
            mask_paths = path_data
        else:
            # Simple format: single path
            mask_paths = [path_data]

        # Load and combine all masks
        combined_mask = None

        for path in mask_paths:
            if not Path(path).exists():
                continue

            if Path(path).suffix == ".npy":
                mask = np.load(path)
            else:
                import cv2
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            # Normalize to [0, 1]
            if mask.max() > 1:
                mask = mask.astype(np.float32) / 255.0

            if combined_mask is None:
                combined_mask = mask.astype(np.float32)
            else:
                combined_mask = np.maximum(combined_mask, mask.astype(np.float32))

        if combined_mask is None:
            # Return empty mask if no valid masks found
            h, w = self.image_size[1], self.image_size[0]
            combined_mask = np.zeros((h, w), dtype=np.float32)

        return combined_mask

    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to image and mask."""
        config = self.augment_config

        # Ensure image is uint8 for augmentation
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Horizontal flip
        if np.random.random() < config["horizontal_flip_prob"]:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Rotation
        if np.random.random() < config["rotation_prob"]:
            angle = np.random.uniform(*config["rotation_range"])
            image, mask = self._rotate(image, mask, angle)

        # Scale
        if np.random.random() < config["scale_prob"]:
            scale = np.random.uniform(*config["scale_range"])
            image, mask = self._scale(image, mask, scale)

        # Brightness
        brightness = np.random.uniform(*config["brightness_range"])
        image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        # Contrast
        contrast = np.random.uniform(*config["contrast_range"])
        mean = image.mean()
        image = np.clip((image.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)

        # Saturation (if color image)
        if image.ndim == 3:
            saturation = np.random.uniform(*config["saturation_range"])
            image = self._adjust_saturation(image, saturation)

        # Gaussian noise
        if np.random.random() < config["noise_prob"]:
            noise_std = np.random.uniform(*config["noise_std_range"])
            noise = np.random.normal(0, noise_std, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Blur
        if np.random.random() < config["blur_prob"]:
            import cv2
            kernel_size = np.random.choice(config["blur_kernel_sizes"])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        return image, mask

    def _rotate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        angle: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate image and mask by given angle."""
        import cv2

        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return image, mask

    def _scale(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale image and mask, keeping original dimensions."""
        import cv2

        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        image_scaled = cv2.resize(image, (new_w, new_h))
        mask_scaled = cv2.resize(mask, (new_w, new_h))

        # Crop or pad to original size
        if scale > 1:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            image = image_scaled[start_h:start_h + h, start_w:start_w + w]
            mask = mask_scaled[start_h:start_h + h, start_w:start_w + w]
        else:
            # Pad with border replication for image, zeros for mask
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2

            image = cv2.copyMakeBorder(
                image_scaled,
                pad_h, h - new_h - pad_h,
                pad_w, w - new_w - pad_w,
                cv2.BORDER_REPLICATE
            )
            mask = cv2.copyMakeBorder(
                mask_scaled,
                pad_h, h - new_h - pad_h,
                pad_w, w - new_w - pad_w,
                cv2.BORDER_CONSTANT,
                value=0
            )

        return image, mask

    def _adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image saturation."""
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def create_dataloaders(
    data_source: Union[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 0,
    augment_train: bool = True,
    augment_val: bool = False,
    seed: int = 42,
    **dataset_kwargs,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation data loaders.

    Args:
        data_source: Path to data directory or (images, masks) tuple
        batch_size: Batch size for both loaders
        val_split: Fraction of data to use for validation (0 for no validation)
        num_workers: Number of data loading workers
        augment_train: Whether to augment training data
        augment_val: Whether to augment validation data
        seed: Random seed for reproducible splits
        **dataset_kwargs: Additional arguments for GateSegmentationDataset

    Returns:
        Tuple of (train_loader, val_loader). val_loader is None if val_split=0
    """
    # Create full dataset
    full_dataset = GateSegmentationDataset(
        data_source=data_source,
        augment=augment_train,
        **dataset_kwargs,
    )

    # Split into train/val
    total_size = len(full_dataset)

    if val_split > 0:
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        # Create validation dataset without augmentation if requested
        if not augment_val:
            val_dataset_no_aug = GateSegmentationDataset(
                data_source=data_source,
                augment=False,
                **dataset_kwargs,
            )
            # Get same indices as split
            val_indices = val_dataset.indices
            val_dataset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
    else:
        train_dataset = full_dataset
        val_dataset = None

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


def create_synthetic_data(
    num_samples: int = 1000,
    image_size: Tuple[int, int] = (64, 48),
    gate_coverage: Tuple[float, float] = (0.05, 0.4),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic training data for testing.

    Generates random images with rectangular "gate" regions.
    Useful for testing the training pipeline.

    Args:
        num_samples: Number of samples to generate
        image_size: (width, height) of images
        gate_coverage: (min, max) fraction of image covered by gate

    Returns:
        Tuple of (images, masks) as numpy arrays
    """
    import cv2

    w, h = image_size
    images = np.zeros((num_samples, h, w, 3), dtype=np.uint8)
    masks = np.zeros((num_samples, h, w), dtype=np.float32)

    for i in range(num_samples):
        # Random background color
        bg_color = np.random.randint(50, 200, 3)
        images[i] = bg_color

        # Random gate color (make it distinct from background)
        gate_color = np.random.randint(0, 255, 3)
        while np.linalg.norm(gate_color - bg_color) < 50:
            gate_color = np.random.randint(0, 255, 3)

        # Random gate size
        coverage = np.random.uniform(*gate_coverage)
        gate_area = int(w * h * coverage)
        aspect_ratio = np.random.uniform(0.5, 2.0)
        gate_h = int(np.sqrt(gate_area / aspect_ratio))
        gate_w = int(gate_area / gate_h)

        # Clamp to image bounds
        gate_h = min(gate_h, h - 2)
        gate_w = min(gate_w, w - 2)

        # Random position
        x = np.random.randint(0, w - gate_w)
        y = np.random.randint(0, h - gate_h)

        # Draw gate on image and mask
        images[i, y:y + gate_h, x:x + gate_w] = gate_color
        masks[i, y:y + gate_h, x:x + gate_w] = 1.0

        # Add some noise to image
        noise = np.random.normal(0, 10, images[i].shape)
        images[i] = np.clip(images[i].astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return images, masks


if __name__ == "__main__":
    # Test the data loader with synthetic data
    print("Testing GateSegmentationDataset with synthetic data...")

    # Create synthetic data
    images, masks = create_synthetic_data(num_samples=100)
    print(f"Created {len(images)} synthetic samples")
    print(f"Image shape: {images[0].shape}, Mask shape: {masks[0].shape}")

    # Create dataset
    dataset = GateSegmentationDataset(
        data_source=(images, masks),
        augment=True,
    )
    print(f"Dataset size: {len(dataset)}")

    # Test __getitem__
    img, mask = dataset[0]
    print(f"Sample image tensor shape: {img.shape}")
    print(f"Sample mask tensor shape: {mask.shape}")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        data_source=(images, masks),
        batch_size=16,
        val_split=0.1,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test a batch
    for batch_img, batch_mask in train_loader:
        print(f"Batch image shape: {batch_img.shape}")
        print(f"Batch mask shape: {batch_mask.shape}")
        print(f"Image range: [{batch_img.min():.3f}, {batch_img.max():.3f}]")
        print(f"Mask range: [{batch_mask.min():.3f}, {batch_mask.max():.3f}]")
        break

    print("\nTest complete!")
