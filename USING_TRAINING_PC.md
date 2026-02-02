# Using the Training PC

Remote training machine (Windows + WSL2 + RTX 5080) accessed via SSH through Tailscale.

## Quick Reference

```bash
# SSH to training PC
ssh ooousay@denis.tail07d7b1.ts.net

# Run WSL command
ssh ooousay@denis.tail07d7b1.ts.net "wsl <command>"

# Check running processes
ssh ooousay@denis.tail07d7b1.ts.net "wsl ps aux" | grep python
```

---

## Setup (One-Time)

### 1. Verify Connection

```bash
ssh ooousay@denis.tail07d7b1.ts.net "echo Connected!"
```

### 2. Tailscale

Both machines must be on the same Tailscale network:
```bash
tailscale status
```

---

## Architecture

```
Local Mac ──SSH──> Windows (training-pc) ──WSL──> Ubuntu
```

**Key paths on training PC (WSL):**
| What | Path |
|------|------|
| Project repo | `/home/ooousay/repos/autonomous-drone-racing/` |
| Models | `models/` |
| Data | `data/` |

---

## Running Training

### 1. Push code from local
```bash
git push
```

### 2. Pull on training PC and run
```bash
ssh ooousay@denis.tail07d7b1.ts.net "wsl bash -c 'cd ~/repos/autonomous-drone-racing && git pull && pip install -e . && python scripts/train_vision_student.py --demos data/teacher_demos --epochs 100'"
```

### Using tmux (for long-running jobs)
```bash
# Start tmux session
ssh ooousay@denis.tail07d7b1.ts.net "wsl tmux new-session -d -s training 'cd ~/repos/autonomous-drone-racing && python scripts/train_vision_student.py --demos data/teacher_demos --epochs 100'"

# Check progress
ssh ooousay@denis.tail07d7b1.ts.net "wsl tmux capture-pane -t training -p" | tail -20

# Kill session
ssh ooousay@denis.tail07d7b1.ts.net "wsl tmux kill-session -t training"
```

---

## File Transfer

### Copy model to local
```bash
scp ooousay@denis.tail07d7b1.ts.net:/home/ooousay/repos/autonomous-drone-racing/models/vision_student/best_model.pt ./models/vision_student/
```

---

## Troubleshooting

### SSH Times Out
1. Check Tailscale: `tailscale status`
2. Ping hostname: `ping denis.tail07d7b1.ts.net`
3. Training PC might be asleep

### WSL Command Fails with "not recognized"
Command is running on Windows CMD, not WSL. Wrap in `wsl bash -c "..."`:
```bash
# Wrong
ssh training-pc "wsl ps aux | grep python"

# Correct (pipe on local)
ssh training-pc "wsl ps aux" | grep python
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

- **No Vulkan** - WSL2 only supports CUDA, not Vulkan
- **Camera rendering works** - gym-pybullet-drones uses OpenGL (CPU) which works fine
- **Isaac Sim abandoned** - Don't use, Vulkan required for cameras
