# rLLM Setup Notes (Winnie)

**Date**: April 12, 2026  
**Goal**: Set up rllm with verl backend for distributed GPU training  
**GPU Node**: h200-137-097-230

## Quick Start (After Setup)

To use rllm on the GPU node:
```bash
ssh h200-137-097-230
cd /home/winnieyangwn/rllm
source .venv/bin/activate
# Now you can use rllm CLI or Python
```

---

## Setup Summary

### Hardware
- **GPUs**: 8x NVIDIA H200 (143GB VRAM each)
- **Driver**: 580.126.09
- **CUDA**: 12.6 (V12.6.85)
- **nvcc**: `/usr/local/cuda/bin/nvcc`

### Software Installed
- **Python**: 3.11.14 (managed by uv in `.venv/`)
- **PyTorch**: 2.10.0+cu128 (CUDA enabled)
- **rllm**: 0.3.0rc0
- **verl**: 0.7.1
- **vLLM**: 0.17.0
- **flash-attn**: 2.8.1
- **ray**: 2.54.1
- **Pre-commit hooks**: Installed

---

## Full Setup Steps (For Replication)

### Prerequisites
- SSH access to GPU node with CUDA
- `uv` installed at `/storage/home/winnieyangwn/.local/bin/uv`
- rllm repo cloned at `/home/winnieyangwn/rllm`

### Step 1: SSH into GPU node
```bash
ssh h200-137-097-230
```

### Step 2: Verify GPU/CUDA availability
```bash
nvidia-smi
nvcc --version
```

### Step 3: Set environment variables
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/storage/home/winnieyangwn/.local/bin:$PATH
```

### Step 4: Install rllm with dev + verl extras
```bash
cd /home/winnieyangwn/rllm
uv sync --extra dev --extra verl
```

**Note**: `uv sync` creates its own `.venv/` directory with Python 3.11. The conda environment is not used for packages - uv manages everything in `.venv/`.

**Build time**: flash-attn takes ~24 minutes to compile on first install.

### Step 5: Install pre-commit hooks
```bash
source .venv/bin/activate
pre-commit install
```

### Step 6: Verify installation
```bash
source .venv/bin/activate
python -c "import rllm; print('rllm: OK')"
python -c "import torch; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import verl; print('verl: OK')"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
```

Expected output:
```
rllm: OK
torch: 2.10.0+cu128, CUDA: True, GPUs: 8
verl: OK
vllm: 0.17.0
```

---

## Important Notes

### Activation
Always use `.venv`, not conda:
```bash
cd /home/winnieyangwn/rllm
source .venv/bin/activate
```

### SSH Sessions
SSH sessions don't source `.bashrc` automatically. If running via `ssh node "command"`, set PATH explicitly:
```bash
ssh h200-137-097-230 "export PATH=/storage/home/winnieyangwn/.local/bin:\$PATH && cd /home/winnieyangwn/rllm && source .venv/bin/activate && python your_script.py"
```

### Lock File Issues
If you see "Could not acquire lock" errors from uv, clear stale locks:
```bash
rm -f ~/.cache/uv/sdists-v9/pypi/flash-attn/2.8.1/.lock
```

---

## Execution Log (April 12, 2026)

1. **SSH Connection**: ✅ Connected to h200-137-097-230
2. **GPU Verification**: ✅ 8x H200 detected, CUDA 12.6 available
3. **Conda env created**: ✅ `rllm` with Python 3.12 (but uv uses its own Python 3.11)
4. **Lock file issue**: ⚠️ Previous attempt left stale lock, cleared with `rm -f .../.lock`
5. **uv sync**: ✅ Installed 295 packages (flash-attn build: 23m 33s)
6. **Pre-commit**: ✅ Hooks installed
7. **Verification**: ✅ All imports successful, CUDA working with 8 GPUs

