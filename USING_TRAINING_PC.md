# Using the Training PC

Remote training machine (Windows + WSL2 + RTX 5080) accessed via SSH through Tailscale.

## Quick Reference

```bash
# Check what's running
./scripts/remote/training-status.sh

# Start training (50K iterations, 4096 envs)
./scripts/remote/start-training.sh 50000 4096

# Monitor training output
./scripts/remote/monitor-training.sh

# Kill all training processes
./scripts/remote/kill-training.sh

# Run any WSL command
./scripts/remote/wsl-run.sh "ps aux | grep python"

# Interactive WSL shell
./scripts/remote/wsl-run.sh
```

---

## Setup (One-Time)

### 1. SSH Config

Add to `~/.ssh/config`:

```
Host training-pc
    HostName denis.tail07d7b1.ts.net
    User ooousay
    ConnectTimeout 15
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### 2. Verify Connection

```bash
ssh training-pc 'echo "Connected!"'
```

### 3. Tailscale

Both machines must be on the same Tailscale network. Check with:
```bash
tailscale status
```

---

## Architecture

```
Local Mac ──SSH──> Windows (training-pc) ──WSL──> Ubuntu (isaac_drone_racer)
```

**Key paths on training PC (WSL):**
| What | Path |
|------|------|
| Isaac Drone Racer | `/home/ooousay/repos/isaac_drone_racer/` |
| Python venv | `/home/ooousay/repos/isaac-racing-venv/` |
| Training logs | `/home/ooousay/repos/isaac_drone_racer/logs/skrl/drone_racer/` |
| Checkpoints | `<run_dir>/checkpoints/` |

---

## Running Commands

### The Quoting Problem

Commands pass through 4 shells: **local → SSH → Windows CMD → WSL → bash**

**Golden rule:** Use single quotes for SSH, double quotes for bash:

```bash
# CORRECT
ssh training-pc 'wsl bash -c "echo hello && pwd"'

# WRONG (quoting breaks)
ssh training-pc "wsl bash -c 'echo hello'"
```

### Simple Commands

```bash
# List files
ssh training-pc 'wsl ls -la /home/ooousay/repos/'

# Check processes
ssh training-pc 'wsl bash -c "ps aux | grep python"'

# Kill a process
ssh training-pc 'wsl bash -c "kill -9 12345"'
```

### Complex Commands (Use Helper Scripts)

For anything with pipes, quotes, or variables, use the helper scripts or write a script on the remote machine.

---

## Training Workflows

### Start Training

```bash
# Default: 50K iterations, 4096 envs
./scripts/remote/start-training.sh

# Custom: 100K iterations, 2048 envs
./scripts/remote/start-training.sh 100000 2048
```

This uses tmux so training survives SSH disconnects.

### Monitor Training

```bash
# Last 50 lines of output
./scripts/remote/monitor-training.sh

# Last 100 lines
./scripts/remote/monitor-training.sh 100
```

### Check Status

```bash
./scripts/remote/training-status.sh
```

Shows: running processes, latest runs, checkpoints, GPU usage.

### Stop Training

```bash
./scripts/remote/kill-training.sh
```

### Attach to Training Session (Interactive)

```bash
ssh -t training-pc 'wsl bash -c "tmux attach -t training"'
```

Press `Ctrl+B, D` to detach without stopping.

---

## File Transfer

### Copy Checkpoint to Local

```bash
# Using scp through WSL
scp training-pc:'$(wsl wslpath -w /home/ooousay/repos/isaac_drone_racer/logs/skrl/drone_racer/2026-02-01_13-18-41_ppo_torch/checkpoints/best_agent.pt)' ./models/
```

### Alternative: Git Push from Remote

```bash
ssh training-pc 'wsl bash -c "cd /home/ooousay/repos/isaac_drone_racer && git add logs/ && git commit -m \"Add checkpoints\" && git push"'
```

---

## Troubleshooting

### SSH Times Out

1. Check Tailscale is running: `tailscale status`
2. Verify hostname resolves: `ping denis.tail07d7b1.ts.net`
3. Training PC might be asleep - wake it up

### WSL Command Fails with "not recognized"

The command is being interpreted by Windows CMD, not WSL. Wrap in `wsl bash -c "..."`:

```bash
# Wrong (grep runs on Windows)
ssh training-pc 'wsl ps aux | grep python'

# Correct (grep runs in WSL)
ssh training-pc 'wsl bash -c "ps aux | grep python"'
```

### Training Not Saving Checkpoints

Check if training is actually running vs stuck:
1. `./scripts/remote/training-status.sh` - look at checkpoint timestamps
2. If no new checkpoints after 30+ min, training may be broken
3. Check logs: `./scripts/remote/monitor-training.sh 100`

### GPU Memory Issues

```bash
# Check GPU memory
ssh training-pc 'nvidia-smi'

# Kill orphaned processes
./scripts/remote/kill-training.sh
```

### Process Won't Die

Use `-9` flag:
```bash
ssh training-pc 'wsl bash -c "kill -9 <PID>"'
```

---

## Hardware Specs

| Component | Spec |
|-----------|------|
| GPU | RTX 5080 (16GB VRAM) |
| RAM | 64GB |
| CPU | 24 cores |
| OS | Windows 11 + WSL2 (Ubuntu) |

### Known Limitations

- **No visual rendering** - WSL2 lacks Vulkan support
- **Isaac Sim 4.5** - RTX 5080 (Blackwell) not officially supported
- **Training only** - evaluation must be headless (`--headless` flag)

---

## Claude Code Integration

When using Claude Code to interact with the training PC:

1. **Always use DNS name**: `denis.tail07d7b1.ts.net` (not IP addresses)
2. **Use helper scripts** when possible (avoids quoting hell)
3. **Check before starting**: Run `./scripts/remote/check-processes.sh` first
4. **Don't spawn unattended agents** that start long-running processes without explicit user approval
