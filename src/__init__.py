"""
AI Grand Prix - Vision-Based Drone Racing

This package implements a vision-based racing architecture for the AI Grand Prix
competition by Anduril. The system uses camera-only perception (no ground truth state)
to control a racing drone through gates.

Components:
- vision: Gate detection and pose estimation from camera images
- state: Extended Kalman Filter for state estimation
- control: Neural network controller and motor mixer
- envs: High-frequency racing environment
- pipeline: End-to-end integration
"""

__version__ = "0.1.0"
