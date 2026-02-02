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

# Check GPU
ssh ooousay@denis.tail07d7b1.ts.net "wsl nvidia-smi"
```

---

## CRITICAL: Running Long Training Jobs

### The Right Way (use scripts, not inline commands)

**Problem:** tmux + SSH + WSL + nested quotes = disaster. Commands fail silently.

**Solution:** Always use a shell script on the remote, never inline commands.

### Step 1: Create/update your training script locally

```bash
# Example: scripts/run_training.sh
#!/bin/bash
set -e  # Exit on error
cd ~/repos/autonomous-drone-racing
source ~/repos/pybullet-venv/bin/activate
export PYTHONUNBUFFERED=1

# Backup existing model
if [ -f models/vision_student/best_model.pt ]; then
    cp models/vision_student/best_model.pt models/vision_student/best_model.backup.pt
    echo "Backed up existing model"
fi

# Run training
python -u scripts/train_vision_student.py \
    --demos data/dart_demos \
    --epochs 100 \
    --batch-size 2048 \
    --num-frames 4 \
    --device cuda \
    2>&1 | tee /tmp/training.log

echo "=== TRAINING COMPLETE ==="
```

### Step 2: Push and pull

```bash
git add scripts/run_training.sh && git commit -m "Update training script" && git push
ssh ooousay@denis.tail07d7b1.ts.net "wsl git -C /home/ooousay/repos/autonomous-drone-racing pull"
```

### Step 3: Start in tmux

```bash
ssh ooousay@denis.tail07d7b1.ts.net "wsl tmux new-session -d -s train /home/ooousay/repos/autonomous-drone-racing/scripts/run_training.sh"
```

### Step 4: Monitor

```bash
# Check tmux output
ssh ooousay@denis.tail07d7b1.ts.net "wsl tmux capture-pane -t train -p" | tail -20

# Check log file
ssh ooousay@denis.tail07d7b1.ts.net "wsl tail -20 /tmp/training.log"

# Check if still running
ssh ooousay@denis.tail07d7b1.ts.net "wsl ps aux" | grep python

# Check GPU utilization
ssh ooousay@denis.tail07d7b1.ts.net "wsl nvidia-smi"
```

---

## Model Management

### ALWAYS backup before retraining

```bash
# Before starting new training
ssh ooousay@denis.tail07d7b1.ts.net "wsl cp /home/ooousay/repos/autonomous-drone-racing/models/vision_student/best_model.pt /home/ooousay/repos/autonomous-drone-racing/models/vision_student/best_model.$(date +%Y%m%d_%H%M%S).pt"
```

### Copy model to local

```bash
scp ooousay@denis.tail07d7b1.ts.net:/home/ooousay/repos/autonomous-drone-racing/models/vision_student/best_model.pt ./models/vision_student/
```

### List saved models

```bash
ssh ooousay@denis.tail07d7b1.ts.net "wsl ls -la /home/ooousay/repos/autonomous-drone-racing/models/vision_student/"
```

---

## Troubleshooting

### No output showing
- Use `python -u` or `export PYTHONUNBUFFERED=1`
- Use `| tee /tmp/training.log` to capture output

### tmux command fails
- **Never use inline commands with quotes**
- Always put commands in a .sh script and run the script

### Training crashed silently
- Check `/tmp/training.log` for errors
- Check `wsl dmesg | tail` for OOM kills

### Model got overwritten
- Should have backed up first (see above)
- Check for `.backup.pt` files

### SSH times out
1. Check Tailscale: `tailscale status`
2. Ping: `ping denis.tail07d7b1.ts.net`
3. PC might be asleep - wake it up

### Pipe commands fail ("grep not recognized")
Pipes run on Windows, not WSL. Do this:
```bash
# Wrong
ssh ... "wsl ps aux | grep python"

# Correct
ssh ... "wsl ps aux" | grep python
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
| Python venv | `/home/ooousay/repos/pybullet-venv/` |
| Models | `models/` |
| Data | `data/` |
| Training logs | `/tmp/training.log` |

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
- **Camera rendering works** - gym-pybullet-drones uses OpenGL (CPU)
- **Isaac Sim abandoned** - Don't use, Vulkan required for cameras
