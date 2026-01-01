# diff-finetune Environment Setup

Environment for DiffSynth-Studio with PyTorch nightly and Flash Attention.

## Virtual Environment

Location: `~/envs/diff-finetune`

Activate:
```bash
source ~/envs/diff-finetune/bin/activate
```

## Installed Packages

### PyTorch (Nightly with CUDA 13.0)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

Installed version: `torch-2.11.0.dev20251231+cu130`

### Flash Attention (Built from Source)

Flash Attention requires source build for Thor GPU (sm_121). Based on: https://github.com/Dao-AILab/flash-attention/issues/1969

#### Build Steps

1. Clone the repository:
```bash
cd /home/duality/projects/DiffSynth-Studio
git clone https://github.com/Dao-AILab/flash-attention.git
```

2. Modify `flash-attention/setup.py` - replace the `add_cuda_gencodes` function with:
```python
def add_cuda_gencodes(cc_flag, archs, bare_metal_version):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_121")
    return cc_flag
```

3. Build and install:
```bash
cd flash-attention
source ~/envs/diff-finetune/bin/activate
pip install ninja packaging

export MAX_JOBS=16
export NVCC_THREADS=2
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS

MAX_JOBS=$MAX_JOBS \
CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS \
FLASH_ATTENTION_FORCE_BUILD="TRUE" \
FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" \
FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" \
pip3 wheel . -v --no-deps --no-build-isolation -w ./wheels/

pip3 install ./wheels/flash_attn*.whl
```

Installed version: `flash_attn-2.8.3`

Pre-built wheel location: `./flash-attention/wheels/flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl`

## Quick Start

```bash
source ~/envs/diff-finetune/bin/activate
cd /home/duality/projects/DiffSynth-Studio
```

## References

- Flash Attention Thor GPU issue: https://github.com/Dao-AILab/flash-attention/issues/1969
- Gencode fix: https://github.com/Dao-AILab/flash-attention/issues/1969#issuecomment-3478145197
- Build flags: https://github.com/Dao-AILab/flash-attention/issues/1969#issuecomment-3543125508
