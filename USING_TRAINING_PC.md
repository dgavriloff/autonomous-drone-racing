# Using the Training PC

Remote training machine (Windows + WSL2 + RTX 5080) via SSH + Tailscale.

---

## 🚨 FOR AGENTS: How to Run Training

**NEVER use inline commands with tmux. ALWAYS use a shell script.**

### Step 1: Edit the training script locally
```bash
# Edit scripts/run_training.sh with your training command
```

### Step 2: Push to remote
```bash
git add -A && git commit -m "Update training" && git push
ssh ooousay@denis.tail07d7b1.ts.net "wsl git -C /home/ooousay/repos/autonomous-drone-racing pull"
```

### Step 3: Start training in tmux
```bash
ssh ooousay@denis.tail07d7b1.ts.net 'wsl tmux new-session -d -s train /home/ooousay/repos/autonomous-drone-racing/scripts/run_training.sh'
```

### Step 4: Launch a monitor subagent
```
Use Task tool with subagent_type="Bash" and run_in_background=true.
Have it check every 2 min:
  ssh ooousay@denis.tail07d7b1.ts.net 'wsl tmux capture-pane -t train -p' | tail -20
Report when "=== ALL DONE ===" appears or process stops.
```

### Step 5: Check status manually (optional)
```bash
# Tmux output
ssh ooousay@denis.tail07d7b1.ts.net 'wsl tmux capture-pane -t train -p' | tail -20

# Process running?
ssh ooousay@denis.tail07d7b1.ts.net "wsl ps aux" | grep python

# GPU usage
ssh ooousay@denis.tail07d7b1.ts.net "wsl nvidia-smi"
```

---

## Common Mistakes (DON'T DO THESE)

❌ `ssh ... "wsl tmux new-session -d -s train 'cd ... && python ...'"`
→ Quoting breaks. Use a script instead.

❌ `ssh ... "wsl ps aux | grep python"`
→ Pipe runs on Windows. Do: `ssh ... "wsl ps aux" | grep python`

❌ Starting training without backing up model
→ The script `run_training.sh` does this automatically.

❌ Not monitoring
→ Training can crash silently. Always launch a monitor subagent.

---

## Key Paths

| What | Path |
|------|------|
| Repo | `/home/ooousay/repos/autonomous-drone-racing/` |
| Venv | `/home/ooousay/repos/pybullet-venv/` |
| Models | `models/vision_student/` |
| Training log | `/tmp/training.log` |

---

## Quick Commands

```bash
# SSH
ssh ooousay@denis.tail07d7b1.ts.net

# Check GPU
ssh ooousay@denis.tail07d7b1.ts.net "wsl nvidia-smi"

# Check processes
ssh ooousay@denis.tail07d7b1.ts.net "wsl ps aux" | grep python

# List tmux sessions
ssh ooousay@denis.tail07d7b1.ts.net 'wsl tmux list-sessions'

# Kill tmux session
ssh ooousay@denis.tail07d7b1.ts.net 'wsl tmux kill-session -t train'

# Copy model to local
scp ooousay@denis.tail07d7b1.ts.net:/home/ooousay/repos/autonomous-drone-racing/models/vision_student/best_model.pt ./models/vision_student/
```

---

## Hardware

- **GPU:** RTX 5080 (16GB VRAM)
- **RAM:** 64GB
- **CPU:** 24 cores
- **OS:** Windows 11 + WSL2 Ubuntu

**Limitations:**
- No Vulkan (CUDA only)
- Isaac Sim doesn't work here
