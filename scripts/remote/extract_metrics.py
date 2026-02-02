#!/usr/bin/env python3
"""Extract metrics from TensorBoard event logs."""
import sys
from tensorboard.backend.event_processing import event_accumulator

def extract_metrics(run_name):
    path = f"/home/ooousay/repos/isaac_drone_racer/logs/skrl/drone_racer/{run_name}"

    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    print(f"=== Metrics for {run_name} ===")
    print()

    # Key metrics we care about
    key_tags = [
        'Reward / Total reward (mean)',
        'Episode / Total timesteps (mean)',
        'Info / Episode_Reward/gate_passed',
        'Info / Episode_Reward/progress',
        'Info / Episode_Reward/terminating',
    ]

    print("Key Metrics:")
    for tag in key_tags:
        try:
            scalars = ea.Scalars(tag)
            if len(scalars) > 0:
                final = scalars[-1].value
                max_val = max(s.value for s in scalars)
                print(f"  {tag}:")
                print(f"    Final: {final:.4f} (step {scalars[-1].step})")
                print(f"    Max:   {max_val:.4f}")
        except KeyError:
            print(f"  {tag}: NOT FOUND")

    print()
    print("All available tags:")
    for t in ea.Tags()["scalars"]:
        print(f"  {t}")

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "2026-02-01_15-15-38_ppo_torch"
    extract_metrics(run)
