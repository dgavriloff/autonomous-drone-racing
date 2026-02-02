#!/usr/bin/env python3
"""
Convert TII Racing dataset to GateNet training format.

The TII Racing dataset provides bounding box labels with corner keypoints.
This script converts those to binary segmentation masks for GateNet training.

Usage:
    python scripts/convert_tii_to_masks.py --input data/tii-racing/data --output data/gatenet_training
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm


def parse_tii_label(label_line: str, img_width: int, img_height: int) -> dict:
    """
    Parse a TII Racing label line.

    Format: 0 cx cy w h tlx tly tlv trx try trv brx bry brv blx bly blv

    Returns dict with:
        - bbox: (cx, cy, w, h) in pixels
        - corners: [(x, y, visible), ...] for tl, tr, br, bl
    """
    parts = label_line.strip().split()
    if len(parts) != 17:
        return None

    class_id = int(parts[0])
    if class_id != 0:  # Only gates
        return None

    # Bounding box (normalized -> pixels)
    cx = float(parts[1]) * img_width
    cy = float(parts[2]) * img_height
    w = float(parts[3]) * img_width
    h = float(parts[4]) * img_height

    # Corner keypoints (normalized -> pixels)
    corners = []
    for i in range(4):
        base_idx = 5 + i * 3
        x = float(parts[base_idx]) * img_width
        y = float(parts[base_idx + 1]) * img_height
        visible = int(parts[base_idx + 2])  # 0=outside, 2=inside
        corners.append((x, y, visible))

    return {
        'bbox': (cx, cy, w, h),
        'corners': corners,  # tl, tr, br, bl
    }


def create_gate_mask(img_width: int, img_height: int, labels: list,
                     use_corners: bool = True, gate_thickness: int = 10) -> np.ndarray:
    """
    Create binary segmentation mask from gate labels.

    Args:
        img_width, img_height: Image dimensions
        labels: List of parsed label dicts
        use_corners: If True, use corner keypoints; else use bbox
        gate_thickness: Thickness of gate frame lines in pixels

    Returns:
        Binary mask (H, W) with gate pixels = 255
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for label in labels:
        if label is None:
            continue

        if use_corners:
            # Use corner keypoints to draw gate frame
            corners = label['corners']
            # Filter visible corners
            visible_corners = [(int(x), int(y)) for x, y, v in corners if v > 0]

            if len(visible_corners) >= 2:
                # Draw lines between consecutive corners
                pts = np.array(visible_corners, dtype=np.int32)
                cv2.polylines(mask, [pts], isClosed=True, color=255,
                             thickness=gate_thickness)
        else:
            # Use bounding box
            cx, cy, w, h = label['bbox']
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, gate_thickness)

    return mask


def process_flight(flight_dir: Path, output_dir: Path,
                   target_size: tuple = (64, 48),
                   gate_thickness: int = 8) -> int:
    """
    Process a single flight directory.

    Returns number of samples processed.
    """
    camera_dir = flight_dir / f"camera_{flight_dir.name}"
    labels_dir = flight_dir / f"label_{flight_dir.name}"  # TII uses "label_" not "labels_"

    if not camera_dir.exists() or not labels_dir.exists():
        print(f"  Skipping {flight_dir.name}: missing camera/labels dir")
        return 0

    # Create output directories
    images_out = output_dir / "images"
    masks_out = output_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    # Process each image
    count = 0
    image_files = sorted(camera_dir.glob("*.jpg")) + sorted(camera_dir.glob("*.jpeg"))

    for img_path in image_files:
        # Find corresponding label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        # Load image
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Parse labels
        with open(label_path, 'r') as f:
            label_lines = f.readlines()

        labels = [parse_tii_label(line, img_width, img_height) for line in label_lines]
        labels = [l for l in labels if l is not None]

        if not labels:
            continue  # Skip images with no gates

        # Create mask
        mask = create_gate_mask(img_width, img_height, labels,
                               use_corners=True, gate_thickness=gate_thickness)

        # Resize to target size
        img_resized = img.resize(target_size, Image.BILINEAR)
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # Save
        sample_name = f"{flight_dir.name}_{img_path.stem}"
        img_resized.save(images_out / f"{sample_name}.png")
        Image.fromarray(mask_resized).save(masks_out / f"{sample_name}.png")

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Convert TII Racing to GateNet format")
    parser.add_argument("--input", type=str, default="data/tii-racing/data",
                       help="TII Racing data directory")
    parser.add_argument("--output", type=str, default="data/gatenet_training",
                       help="Output directory for GateNet training data")
    parser.add_argument("--width", type=int, default=64, help="Target image width")
    parser.add_argument("--height", type=int, default=48, help="Target image height")
    parser.add_argument("--thickness", type=int, default=8,
                       help="Gate frame thickness in mask (pixels at original resolution)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    target_size = (args.width, args.height)

    print(f"Converting TII Racing dataset")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Target size: {target_size}")

    total_samples = 0

    # Process autonomous flights
    auto_dir = input_dir / "autonomous"
    if auto_dir.exists():
        print(f"\nProcessing autonomous flights...")
        for flight_dir in sorted(auto_dir.iterdir()):
            if flight_dir.is_dir() and flight_dir.name.startswith("flight-"):
                count = process_flight(flight_dir, output_dir, target_size, args.thickness)
                print(f"  {flight_dir.name}: {count} samples")
                total_samples += count

    # Process piloted flights
    pilot_dir = input_dir / "piloted"
    if pilot_dir.exists():
        print(f"\nProcessing piloted flights...")
        for flight_dir in sorted(pilot_dir.iterdir()):
            if flight_dir.is_dir() and flight_dir.name.startswith("flight-"):
                count = process_flight(flight_dir, output_dir, target_size, args.thickness)
                print(f"  {flight_dir.name}: {count} samples")
                total_samples += count

    print(f"\nTotal samples: {total_samples}")
    print(f"Output saved to: {output_dir}")

    # Save metadata
    metadata = {
        'source': 'TII Racing Dataset',
        'total_samples': total_samples,
        'image_size': list(target_size),
        'gate_thickness': args.thickness,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
