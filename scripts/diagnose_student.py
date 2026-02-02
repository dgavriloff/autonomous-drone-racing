#!/usr/bin/env python3
"""
Diagnose why vision student fails.
1. Track which gates it misses
2. Measure how close misses are
3. Compare student vs teacher actions
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from scripts.train_vision_student import VisionStudentNetV2
from scripts.collect_teacher_demos import CameraRacingEnv
from scripts.train_parallel import create_simple_track


def diagnose(student_path, teacher_path, num_episodes=10, device="cuda"):
    print("=" * 60)
    print("DIAGNOSING VISION STUDENT")
    print("=" * 60)

    # Load student
    checkpoint = torch.load(student_path, map_location=device, weights_only=False)
    num_frames = checkpoint.get("num_frames", 4)
    student = VisionStudentNetV2(
        action_dim=4, hidden_dims=(256, 256), device=device, num_frames=num_frames
    ).to(device)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()
    print(f"Loaded student (num_frames={num_frames})")

    # Load teacher
    teacher = SAC.load(teacher_path)
    print(f"Loaded teacher")

    # Create env
    track = create_simple_track(num_gates=5, radius=1.5)
    env = CameraRacingEnv(
        track=track, image_size=(64, 48), ctrl_freq=48, pyb_freq=240,
        gui=False, gate_tolerance=1.0, max_steps=1000
    )

    # Track statistics
    gate_fail_counts = {i: 0 for i in range(5)}
    gate_pass_counts = {i: 0 for i in range(5)}
    action_diffs = []
    miss_distances = []

    print(f"\nRunning {num_episodes} episodes...")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        frame_buffer = []
        prev_gate = 0
        ep_action_diffs = []

        while not done:
            # Get camera image
            rgb = env.get_camera_image()
            rgb_norm = rgb.astype(np.float32) / 255.0
            frame_buffer.append(rgb_norm)
            if len(frame_buffer) > num_frames:
                frame_buffer.pop(0)

            # Pad if needed
            if len(frame_buffer) < num_frames:
                padded = [frame_buffer[0]] * (num_frames - len(frame_buffer)) + frame_buffer
            else:
                padded = frame_buffer

            # Get student action
            stacked = np.concatenate(padded, axis=-1)
            img_tensor = torch.from_numpy(stacked).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                student_action = student(img_tensor).squeeze().cpu().numpy()

            # Get teacher action (what it WOULD do)
            teacher_action, _ = teacher.predict(obs, deterministic=True)
            teacher_action = teacher_action.squeeze()

            # Compare actions
            action_diff = np.abs(student_action - teacher_action).mean()
            ep_action_diffs.append(action_diff)

            # Step with student action
            obs, reward, terminated, truncated, info = env.step(student_action.reshape(1, -1))
            done = terminated or truncated

            # Track gate progress
            current_gate = info.get("gates_passed", 0)
            if current_gate > prev_gate:
                gate_pass_counts[prev_gate] += 1
                prev_gate = current_gate

        # Record which gate failed
        final_gates = info.get("gates_passed", 0)
        if final_gates < 5:
            gate_fail_counts[final_gates] += 1

            # Get distance to missed gate
            if hasattr(env, 'current_gate_idx') and hasattr(env, 'track'):
                try:
                    drone_pos = obs[:3] if len(obs) >= 3 else [0, 0, 0]
                    gate_pos = env.track.gates[final_gates].position
                    miss_dist = np.linalg.norm(np.array(drone_pos) - np.array(gate_pos))
                    miss_distances.append(miss_dist)
                except:
                    pass

        action_diffs.extend(ep_action_diffs)
        print(f"  Ep {ep+1}: {final_gates}/5 gates, avg action diff: {np.mean(ep_action_diffs):.3f}")

    env.close()

    # Analysis
    print("\n" + "=" * 60)
    print("DIAGNOSIS RESULTS")
    print("=" * 60)

    print("\n1. WHICH GATES FAIL?")
    for i in range(5):
        fail = gate_fail_counts[i]
        passed = gate_pass_counts[i]
        total = fail + passed if (fail + passed) > 0 else 1
        print(f"   Gate {i+1}: {passed}/{total} passed ({100*passed/total:.0f}%), failed {fail} times")

    print("\n2. ACTION DIFFERENCE (student vs teacher)")
    print(f"   Mean: {np.mean(action_diffs):.4f}")
    print(f"   Std:  {np.std(action_diffs):.4f}")
    print(f"   Max:  {np.max(action_diffs):.4f}")
    if np.mean(action_diffs) < 0.1:
        print("   → Actions are SIMILAR - precision/timing issue (RL should help)")
    elif np.mean(action_diffs) < 0.3:
        print("   → Actions are MODERATELY different - partial understanding")
    else:
        print("   → Actions are VERY different - vision/policy issue (RL may not help)")

    if miss_distances:
        print("\n3. MISS DISTANCES")
        print(f"   Mean: {np.mean(miss_distances):.2f}m")
        print(f"   Min:  {np.min(miss_distances):.2f}m")
        if np.mean(miss_distances) < 0.5:
            print("   → Close misses - precision issue (RL should help)")
        else:
            print("   → Far misses - direction/vision issue")

    print("\n" + "=" * 60)
    return gate_fail_counts, np.mean(action_diffs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="models/vision_student/best_model.pt")
    parser.add_argument("--teacher", default="models/curriculum_final.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    diagnose(args.student, args.teacher, args.episodes, args.device)
