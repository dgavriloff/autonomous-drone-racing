#!/usr/bin/env python3
"""
Export trajectory data from a trained IsaacLab drone racing policy.
Runs headless and outputs CSV for visualization.

Usage:
    python export_trajectory.py --checkpoint path/to/agent.pt --output trajectory.csv --episodes 5
"""

import argparse
import csv
import sys

# Parse args before importing heavy modules
parser = argparse.ArgumentParser(description="Export trajectory from trained policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent checkpoint (.pt)")
parser.add_argument("--output", type=str, default="trajectory.csv", help="Output CSV file")
parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to record")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (use 1 for clean trajectories)")

# Add IsaacLab launcher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force headless mode and CPU device
args.headless = True
args.enable_cameras = False
args.device = "cpu"  # Use CPU to avoid GPU conflicts

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import the rest
import torch
import torch.nn as nn

# Import environment and config
from isaaclab.envs import ManagerBasedRLEnv
from tasks.drone_racer.drone_racer_env_cfg import DroneRacerEnvCfg_PLAY


class SharedModel(nn.Module):
    """Model matching skrl's SharedModel architecture from training."""
    
    def __init__(self, obs_dim, action_dim, device):
        super().__init__()
        self.device = device
        
        # Architecture from skrl config: 256, 256, 256 with ELU
        self.net_container = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        self.policy_layer = nn.Linear(256, action_dim)
        self.value_layer = nn.Linear(256, 1)
        self.log_std_parameter = nn.Parameter(torch.zeros(action_dim))
        
        self.to(device)
    
    def get_action(self, obs):
        """Get deterministic action (mean of Gaussian)."""
        with torch.no_grad():
            net_output = self.net_container(obs)
            action = self.policy_layer(net_output)
        return action


def load_policy(checkpoint_path: str, obs_dim: int, action_dim: int, device):
    """Load trained policy from checkpoint."""
    model = SharedModel(obs_dim, action_dim, device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load only the policy weights (not value or optimizer)
    policy_state = checkpoint.get("policy", checkpoint)
    model.load_state_dict(policy_state, strict=True)
    model.eval()
    
    return model


def export_trajectory(checkpoint_path: str, output_path: str, num_episodes: int, num_envs: int):
    """Run policy and export trajectory to CSV."""
    
    print(f"Loading environment...")
    
    # Create environment config
    env_cfg = DroneRacerEnvCfg_PLAY()
    env_cfg.scene.num_envs = num_envs
    
    # Create environment directly
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Debug: print action space structure
    print(f"Action space type: {type(env.action_space)}")
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    
    # Get dimensions by doing a reset and checking the tensor shapes
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    obs_dim = obs.shape[1]  # (num_envs, obs_dim)
    
    # Action space is Box with shape (num_envs, action_dim) for vectorized envs
    # Take the last dimension as action_dim
    action_shape = env.action_space.shape
    action_dim = action_shape[-1] if len(action_shape) > 1 else action_shape[0]
    
    device = env.device
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model = load_policy(checkpoint_path, obs_dim, action_dim, device)
    
    print(f"Recording {num_episodes} episodes...")
    
    # CSV header
    fieldnames = ["episode", "step", "timestamp", "x", "y", "z", "qw", "qx", "qy", "qz",
                  "vx", "vy", "vz", "wx", "wy", "wz", "target_x", "target_y", "target_z", "reward"]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}/{num_episodes}")
            obs_dict, info = env.reset()
            obs = obs_dict["policy"]  # Get policy observations
            
            done = False
            step = 0
            total_reward = 0
            
            while not done and simulation_app.is_running():
                # Get action from policy
                action = model.get_action(obs)
                
                # Step environment
                next_obs_dict, reward, terminated, truncated, info = env.step(action)
                next_obs = next_obs_dict["policy"]
                done = terminated.any() or truncated.any()
                
                # Parse observation: [pos(3), quat(4), lin_vel(3), ang_vel(3), target(3), last_action(4)] = 20
                # Note: lin_vel and ang_vel are in body frame
                obs_np = obs[0].cpu().numpy()
                pos = obs_np[0:3]
                quat = obs_np[3:7]  # w, x, y, z
                lin_vel = obs_np[7:10]
                ang_vel = obs_np[10:13]
                target_pos = obs_np[13:16]
                
                # Write row
                writer.writerow({
                    "episode": episode,
                    "step": step,
                    "timestamp": step * 0.01,  # dt = 1/400, decimation = 4 -> 100Hz
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                    "qw": float(quat[0]),
                    "qx": float(quat[1]),
                    "qy": float(quat[2]),
                    "qz": float(quat[3]),
                    "vx": float(lin_vel[0]),
                    "vy": float(lin_vel[1]),
                    "vz": float(lin_vel[2]),
                    "wx": float(ang_vel[0]),
                    "wy": float(ang_vel[1]),
                    "wz": float(ang_vel[2]),
                    "target_x": float(target_pos[0]),
                    "target_y": float(target_pos[1]),
                    "target_z": float(target_pos[2]),
                    "reward": float(reward[0].cpu().numpy()),
                })
                
                total_reward += float(reward[0].cpu().numpy())
                obs = next_obs
                step += 1
                
                # Safety limit
                if step > 2000:
                    print(f"    Step limit reached")
                    break
            
            print(f"    Steps: {step}, Reward: {total_reward:.2f}")
    
    print(f"Trajectory saved to: {output_path}")
    env.close()


if __name__ == "__main__":
    export_trajectory(args.checkpoint, args.output, args.episodes, args.num_envs)
    simulation_app.close()
