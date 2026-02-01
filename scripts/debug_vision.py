#!/usr/bin/env python3
"""Debug the vision pipeline by visualizing camera output."""

import sys
from pathlib import Path
import numpy as np
import torch
import pybullet as p
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_parallel import VelocityRacingEnv, create_simple_track
from src.vision.gate_net import create_gatenet
from src.vision.quad_gate import QuAdGate, visualize_detection


def create_gate_visuals(track, physics_client=0):
    """Create visual markers for gates in PyBullet."""
    gate_ids = []
    for gate in track.gates:
        # Create gate frame visualization - orange box
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.05, gate.width/2, gate.height/2],
            rgbaColor=[1, 0.5, 0, 1.0],  # Orange, fully opaque
            physicsClientId=physics_client,
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=gate.position,
            baseOrientation=gate.orientation,
            physicsClientId=physics_client,
        )
        gate_ids.append(body_id)
    return gate_ids


def debug_camera():
    """Debug camera capture and vision pipeline."""
    print("Setting up environment...")

    # Create track and env with GUI
    track = create_simple_track(num_gates=5, radius=1.5)
    env = VelocityRacingEnv(track, gui=True, max_steps=100)

    # Load GateNet
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    gatenet = create_gatenet("models/gate_net/best_model.pt", device=device)
    quad_gate = QuAdGate()

    # Reset first
    obs, _ = env.reset()

    # Create gate visuals AFTER reset (to ensure physics client is ready)
    print("Creating gate visuals...")
    gate_ids = create_gate_visuals(track, env.CLIENT)
    print(f"Gate IDs created: {gate_ids}")

    # Verify gates exist
    for i, gid in enumerate(gate_ids):
        try:
            gpos, gorn = p.getBasePositionAndOrientation(gid, physicsClientId=env.CLIENT)
            print(f"Gate {i} body verified: pos={gpos}")
        except Exception as e:
            print(f"Gate {i} body FAILED: {e}")

    # Position drone 2m away from gate 0
    gate_pos = track.gates[0].position
    drone_start = gate_pos - np.array([2.0, 0, 0])  # 2m behind gate 0

    # Set drone position
    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0],
        drone_start,
        [0, 0, 0, 1],  # Facing +X
        physicsClientId=env.CLIENT,
    )

    # Run a few steps to stabilize
    for i in range(10):
        action = np.array([[0, 0, 0, 0]])  # Hover
        obs, _, _, _, info = env.step(action)

    # Get drone state
    state = env._getDroneStateVector(0)
    pos = state[0:3]
    quat = state[3:7]

    print(f"Drone position: {pos}")
    print(f"Drone orientation (quat): {quat}")

    # Compute camera parameters
    image_size = (64, 48)
    camera_fov = 60.0

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=camera_fov,
        aspect=image_size[0] / image_size[1],
        nearVal=0.01,
        farVal=100.0,
    )

    # Camera position and orientation
    rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    forward = rot_matrix @ np.array([1, 0, 0])  # x-forward
    up = rot_matrix @ np.array([0, 0, 1])
    camera_target = pos + forward

    print(f"Camera forward: {forward}")
    print(f"Camera target: {camera_target}")

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=pos,
        cameraTargetPosition=camera_target,
        cameraUpVector=up,
    )

    # Capture image
    width, height = image_size
    _, _, rgb, depth, seg = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=env.CLIENT,  # Important!
    )

    rgb = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    print(f"Captured image shape: {rgb.shape}")
    print(f"Image mean: {rgb.mean():.1f}, min: {rgb.min()}, max: {rgb.max()}")

    # Save raw image
    cv2.imwrite("/tmp/debug_camera_raw.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("Saved raw image to /tmp/debug_camera_raw.png")

    # Run GateNet
    img_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        mask = gatenet(img_tensor)

    mask_np = mask.squeeze().cpu().numpy()
    print(f"Mask shape: {mask_np.shape}")
    print(f"Mask mean: {mask_np.mean():.4f}, min: {mask_np.min():.4f}, max: {mask_np.max():.4f}")

    # Save mask
    mask_vis = (mask_np * 255).astype(np.uint8)
    cv2.imwrite("/tmp/debug_mask.png", mask_vis)
    print("Saved mask to /tmp/debug_mask.png")

    # Run QuAdGate
    detection = quad_gate.detect(mask_np)

    if detection is not None:
        print(f"Detection: confidence={detection.confidence:.3f}, corners={detection.corners}")
        vis = visualize_detection(rgb, detection, mask_np)
        cv2.imwrite("/tmp/debug_detection.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print("Saved detection visualization to /tmp/debug_detection.png")
    else:
        print("No gate detected!")

    # Also check what gates look like in the scene
    print("\nGate positions:")
    for i, gate in enumerate(track.gates):
        dist = np.linalg.norm(gate.position - pos)
        direction = (gate.position - pos) / dist
        angle = np.arccos(np.dot(forward, direction))
        print(f"  Gate {i}: pos={gate.position}, dist={dist:.2f}m, angle={np.degrees(angle):.1f}°")

    env.close()
    print("\nDone! Check /tmp/debug_*.png files")


if __name__ == "__main__":
    debug_camera()
