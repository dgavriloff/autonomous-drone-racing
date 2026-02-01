#!/usr/bin/env python3
"""
Test RACE drone stability with the updated PID controller.

Verifies that roll/pitch remain stable under velocity commands.
"""

import sys
from pathlib import Path
import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from src.control.race_pid_control import RacePIDControl


def test_race_stability(duration=5.0, gui=False):
    """Test RACE drone hover stability."""
    print("=" * 60)
    print("RACE DRONE STABILITY TEST")
    print("=" * 60)
    print(f"Duration: {duration}s")
    print(f"Using: RacePIDControl with inertia_ratio=180")
    print()

    # Create environment with RACE drone
    env = CtrlAviary(
        drone_model=DroneModel.RACE,
        num_drones=1,
        initial_xyzs=np.array([[0, 0, 0.5]]),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=48,
        gui=gui,
    )

    # Create controller
    ctrl = RacePIDControl(drone_model=DroneModel.RACE, g=env.G)

    print(f"RACE drone specs:")
    print(f"  Mass: {env.M:.3f} kg")
    print(f"  KF: {env.KF:.2e}")
    print(f"  Inertia: {env.J}")
    print(f"  MAX_SPEED_KMH: {env.MAX_SPEED_KMH}")
    print()

    # Target: hover at starting position
    target_pos = np.array([0, 0, 0.5])
    target_vel = np.zeros(3)
    target_rpy = np.zeros(3)

    steps = int(duration * env.CTRL_FREQ)
    max_roll = 0.0
    max_pitch = 0.0
    roll_history = []
    pitch_history = []
    pos_history = []

    print(f"Running {steps} control steps...")
    print()

    obs, info = env.reset()

    for step in range(steps):
        # Get state
        state = env._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        vel = state[10:13]
        ang_vel = state[13:16]
        rpy = np.array(p.getEulerFromQuaternion(quat))

        # Track extremes
        roll_history.append(abs(rpy[0]))
        pitch_history.append(abs(rpy[1]))
        pos_history.append(pos.copy())
        max_roll = max(max_roll, abs(rpy[0]))
        max_pitch = max(max_pitch, abs(rpy[1]))

        # Check for instability
        if abs(rpy[0]) > 1.2 or abs(rpy[1]) > 1.2:
            print(f"FAILED at step {step}: roll={rpy[0]:.3f}, pitch={rpy[1]:.3f}")
            print(f"  Position: {pos}")
            break

        # Compute control
        rpm, pos_e, yaw_e = ctrl.computeControl(
            control_timestep=1.0 / env.CTRL_FREQ,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=ang_vel,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
        )

        # Apply control
        env.step(rpm.reshape(1, 4))

        # Progress
        if (step + 1) % (env.CTRL_FREQ) == 0:
            t = (step + 1) / env.CTRL_FREQ
            print(f"t={t:.1f}s: pos={pos}, roll={rpy[0]:.4f}, pitch={rpy[1]:.4f}")

    env.close()

    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Max roll:  {max_roll:.4f} rad ({np.degrees(max_roll):.2f}°)")
    print(f"Max pitch: {max_pitch:.4f} rad ({np.degrees(max_pitch):.2f}°)")
    print(f"Avg roll:  {np.mean(roll_history):.4f} rad")
    print(f"Avg pitch: {np.mean(pitch_history):.4f} rad")

    # Check success criteria
    if max_roll < 0.3 and max_pitch < 0.3:  # Less than ~17 degrees
        print()
        print("✓ STABILITY TEST PASSED!")
        print("  Roll and pitch remained within acceptable bounds.")
        return True
    else:
        print()
        print("✗ STABILITY TEST FAILED")
        print("  Roll or pitch exceeded 0.3 rad (17°)")
        return False


def test_race_velocity(duration=5.0, target_vel_xy=1.0, gui=False):
    """Test RACE drone stability under forward velocity command."""
    print()
    print("=" * 60)
    print(f"RACE DRONE VELOCITY TEST ({target_vel_xy} m/s)")
    print("=" * 60)

    env = CtrlAviary(
        drone_model=DroneModel.RACE,
        num_drones=1,
        initial_xyzs=np.array([[0, 0, 0.5]]),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=48,
        gui=gui,
    )

    ctrl = RacePIDControl(drone_model=DroneModel.RACE, g=env.G)

    # Moving target: fly in +X direction
    target_rpy = np.zeros(3)

    steps = int(duration * env.CTRL_FREQ)
    max_roll = 0.0
    max_pitch = 0.0
    max_speed = 0.0
    pos_history = []

    obs, info = env.reset()

    for step in range(steps):
        state = env._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        vel = state[10:13]
        ang_vel = state[13:16]
        rpy = np.array(p.getEulerFromQuaternion(quat))

        speed = np.linalg.norm(vel[:2])
        max_roll = max(max_roll, abs(rpy[0]))
        max_pitch = max(max_pitch, abs(rpy[1]))
        max_speed = max(max_speed, speed)
        pos_history.append(pos.copy())

        # Check for instability
        if abs(rpy[0]) > 1.2 or abs(rpy[1]) > 1.2:
            print(f"FAILED at step {step}: roll={rpy[0]:.3f}, pitch={rpy[1]:.3f}")
            break

        # Target: current position + velocity direction
        target_pos = pos + np.array([target_vel_xy * 0.5, 0, 0])
        target_pos[2] = 0.5  # Maintain altitude
        target_vel = np.array([target_vel_xy, 0, 0])

        rpm, pos_e, yaw_e = ctrl.computeControl(
            control_timestep=1.0 / env.CTRL_FREQ,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=ang_vel,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
        )

        env.step(rpm.reshape(1, 4))

        if (step + 1) % (env.CTRL_FREQ) == 0:
            t = (step + 1) / env.CTRL_FREQ
            print(f"t={t:.1f}s: pos_x={pos[0]:.2f}m, speed={speed:.2f}m/s, roll={rpy[0]:.4f}")

    env.close()

    print()
    print(f"Max speed achieved: {max_speed:.2f} m/s")
    print(f"Max roll:  {max_roll:.4f} rad ({np.degrees(max_roll):.2f}°)")
    print(f"Max pitch: {max_pitch:.4f} rad ({np.degrees(max_pitch):.2f}°)")
    print(f"Final X position: {pos_history[-1][0]:.2f}m")

    if max_roll < 0.5 and max_pitch < 0.5:
        print("✓ VELOCITY TEST PASSED!")
        return True
    else:
        print("✗ VELOCITY TEST FAILED")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration in seconds")
    parser.add_argument("--velocity", type=float, default=2.0, help="Target velocity for velocity test")
    args = parser.parse_args()

    # Test 1: Hover stability
    hover_ok = test_race_stability(duration=args.duration, gui=args.gui)

    # Test 2: Forward velocity
    if hover_ok:
        velocity_ok = test_race_velocity(
            duration=args.duration,
            target_vel_xy=args.velocity,
            gui=args.gui
        )

        if hover_ok and velocity_ok:
            print()
            print("=" * 60)
            print("ALL TESTS PASSED - RACE drone is stable!")
            print("Ready to update VelocityRacingEnv to use RACE drone.")
            print("=" * 60)
