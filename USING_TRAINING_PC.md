# Using the Training PC

Remote: Windows + WSL2 + RTX 5080 (16GB), 64GB RAM, 24 cores.

---

## ⚠️ USE THE HARDWARE PROPERLY

**Before running ANY training/eval script, verify:**
- `SubprocVecEnv` with 16 envs (NOT single env)
- `num_workers=4` in DataLoader (NOT 0)
- `batch_size=2048+` for GPU efficiency
- `device="cuda"` (NOT cpu)

Single-threaded = wasting 90% of the machine. Always parallelize.

---

## How to Run Training

**NEVER use inline tmux commands. ALWAYS use a shell script.**

### Step 1: Push code
```bash
git add -A && git commit -m "Update" && git push
ssh ooousay@denis.tail07d7b1.ts.net "wsl git -C /home/ooousay/repos/autonomous-drone-racing pull"
```

### Step 2: Start training
```bash
ssh ooousay@denis.tail07d7b1.ts.net 'wsl tmux new-session -d -s train /home/ooousay/repos/autonomous-drone-racing/scripts/run_training.sh'
```

### Step 3: Monitor
```bash
ssh ooousay@denis.tail07d7b1.ts.net 'wsl tmux capture-pane -t train -p' | tail -20
ssh ooousay@denis.tail07d7b1.ts.net "wsl ps aux" | grep python
```

---

## Common Mistakes

❌ **Single-threaded collection** → Use `SubprocVecEnv([make_env] * 16)`
❌ **Inline tmux commands** → Quoting breaks. Use shell scripts.
❌ **`"wsl ps | grep"`** → Pipe on Windows. Do: `"wsl ps" | grep`
❌ **BC loss for model selection** → Eval on gates passed, not val_loss.

---

## Key Paths

| What | Path |
|------|------|
| Repo | `/home/ooousay/repos/autonomous-drone-racing/` |
| Venv | `/home/ooousay/repos/pybullet-venv/` |
| Models | `models/vision_student/` |
