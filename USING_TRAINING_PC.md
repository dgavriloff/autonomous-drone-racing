# Using the Training PC

Remote training machine accessed via SSH through Tailscale.

## Connection

```bash
# Connect to WSL on the training PC
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "cd ~/repos && exec bash"'

# Or run commands directly (recommended for scripts)
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "YOUR_COMMAND_HERE"'
```

## Important Notes

### Quoting
The SSH command goes through multiple layers: local shell → SSH → Windows → WSL → bash.
- Use **single quotes** for the outer SSH command
- Use **double quotes** for the inner bash -c command

```bash
# Correct
ssh ... 'wsl -e bash -c "echo hello && pwd"'

# Wrong (quoting issues)
ssh ... "wsl -e bash -c 'echo hello && pwd'"
```

### Environment Location
```
~/repos/autonomous-drone-racing/
├── venv/           # Python virtual environment
├── scripts/        # Training scripts
├── src/            # Source code
└── models/         # Saved models
```

### Activating the Environment
Always activate venv before running Python:
```bash
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "cd ~/repos/autonomous-drone-racing && source venv/bin/activate && python YOUR_SCRIPT.py"'
```

### System Setup (already done)
The following packages were installed:
- `python3.12-venv` (via apt)
- `build-essential python3-dev` (for compiling pybullet)
- Python packages: torch, stable-baselines3, gymnasium, pybullet, gym-pybullet-drones

### GPU Available
The machine has CUDA GPU support. PyTorch automatically detects and uses it ("Using cuda device").

## Running Training

### Short run (foreground)
```bash
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "cd ~/repos/autonomous-drone-racing && source venv/bin/activate && python scripts/train_velocity_control.py --train --timesteps 200000"'
```

### Long run (use tmux or nohup)
For long training runs, use tmux to prevent disconnection issues:
```bash
# Start tmux session
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "tmux new-session -d -s training \"cd ~/repos/autonomous-drone-racing && source venv/bin/activate && python scripts/train_velocity_control.py --train --timesteps 1000000\""'

# Check on training
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "tmux capture-pane -t training -p | tail -50"'

# Attach to session (interactive)
ssh -t ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "tmux attach -t training"'
```

## Syncing Code

### Push from local
```bash
git add . && git commit -m "message" && git push
```

### Pull on remote
```bash
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "cd ~/repos/autonomous-drone-racing && git pull"'
```

## Retrieving Models

### Copy model to local
```bash
scp ooousay@denis.tail07d7b1.ts.net:'wsl -e cat ~/repos/autonomous-drone-racing/models/velocity_control/final.zip' ./models/
```

Or commit and push from remote:
```bash
ssh ooousay@denis.tail07d7b1.ts.net 'wsl -e bash -c "cd ~/repos/autonomous-drone-racing && git add models/ && git commit -m \"Add trained model\" && git push"'
```

## Troubleshooting

### apt lock errors
Previous apt process may be running:
```bash
ssh ... 'wsl -e bash -c "sudo killall apt apt-get; sudo rm -f /var/lib/apt/lists/lock"'
```

### pip not found
Use `python3 -m pip` instead of `pip3`

### pybullet build fails
Install build tools first:
```bash
ssh ... 'wsl -e bash -c "sudo apt install -y build-essential python3-dev"'
```
