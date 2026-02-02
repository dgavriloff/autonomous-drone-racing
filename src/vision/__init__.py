"""Vision module for gate detection and pose estimation."""

# Lazy import for DataCollector (requires pybullet)
def get_data_collector():
    from .data_collector import DataCollector
    return DataCollector

from .data_loader import GateSegmentationDataset, create_dataloaders, create_synthetic_data
from .gate_net import (
    GateNet,
    GateNetTrainer,
    SegmentationMetrics,
    DiceLoss,
    FocalLoss,
    CombinedLoss,
    create_gatenet,
)
from .quad_gate import QuAdGate
from .pose_estimator import PoseEstimator

__all__ = [
    # Data collection and loading
    "DataCollector",
    "GateSegmentationDataset",
    "create_dataloaders",
    "create_synthetic_data",
    # GateNet model and training
    "GateNet",
    "GateNetTrainer",
    "SegmentationMetrics",
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    "create_gatenet",
    # Corner detection and pose estimation
    "QuAdGate",
    "PoseEstimator",
]
