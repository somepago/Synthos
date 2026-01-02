# Synthos Project Context

## Project Overview

**Synthos-I2I**: CLIP-conditioned image generation using cross-attention conditioning only (no VAE latent initialization). Generation starts from pure noise and is guided by CLIP image embeddings + text prompts.

## Repository

- **GitHub**: https://github.com/somepago/Synthos
- **Local Path**: `/home/duality/projects/DiffSynth-Studio`
- **Branch**: main

## Environment

- **Python Environment**: `~/envs/diff-finetune/bin/activate`
- **Wandb**: Logged in (API key stored)
- **HuggingFace**: Logged in (token stored)

## Key Files

### Training
- **Training Script**: `examples/wanvideo/model_training/full/Synthos-I2I.sh`
- **Training Module**: `examples/wanvideo/model_training/train.py`
- **Runner**: `diffsynth/diffusion/runner.py` (with wandb + validation support)
- **Parsers**: `diffsynth/diffusion/parsers.py`

### Inference
- **Inference Script**: `examples/wanvideo/model_inference/Synthos-I2I.py`

### Data
- **Dataset (small)**: `data/contrastyles/` (84 images from ContraStyles)
- **Dataset (full)**: `data/contrastyles_full/` (363k images, 135k after filtering)
- **Filtered Metadata**: `data/contrastyles_full/metadata_filtered.csv` (min_dim >= 512)
- **Filter Script**: `scripts/filter_metadata_by_resolution.py`

## Model Architecture

```
Synthos-I2I
├── DiT backbone (from Wan2.1-T2V-1.3B)
├── T5 text encoder
├── CLIP image encoder (cross-attention conditioning)
└── VAE (output decoding only)
```

Key settings:
- `dit.has_clip_input = True`
- `dit.require_vae_embedding = False`

## Training Configuration

**Resolution**: 480x832 (matches WAN pretraining, ~16:9 aspect ratio)

```bash
# Full dataset training (img_emb only)
accelerate launch --mixed_precision bf16 examples/wanvideo/model_training/train.py \
  --dataset_base_path data/contrastyles_full \
  --dataset_metadata_path data/contrastyles_full/metadata_filtered.csv \
  --height 480 --width 832 --num_frames 1 \
  --train_img_emb_only --i2i_mode \
  ...
```

## Features Implemented

1. **Wandb Integration**: Training logs loss, learning rate, and validation images to wandb
2. **Batch Size**: Configurable via `--batch_size`
3. **Mixed Precision**: bf16 via accelerate
4. **Conditioning Dropout**: `--text_dropout_prob` and `--image_dropout_prob` for CFG training
5. **Validation During Training**: `--validate_steps`, `--validation_prompts`, `--validation_images`
6. **CLIP-only Conditioning**: No VAE latent initialization, pure cross-attention guidance
7. **Train img_emb Only Mode**: `--train_img_emb_only` flag to freeze DiT except img_emb layer

## Train img_emb Only Mode

Added `--train_img_emb_only` flag to train only the CLIP projection layer (`img_emb`) while freezing all other DiT parameters.

**Script**: `examples/wanvideo/model_training/full/Synthos-I2I-img-emb-only.sh`

**img_emb Architecture** (MLP: 1280 → 1536):
- LayerNorm(1280): ~2.5K params
- Linear(1280→1280): ~1.6M params
- GELU
- Linear(1280→1536): ~2.0M params
- LayerNorm(1536): ~3K params
- **Total: ~3.6M params** (0.3% of 1.3B DiT)

**Optimizations implemented**:
1. Disabled gradient checkpointing (no activation recomputation)
2. Wrapped frozen encoder units (text/CLIP) in `torch.no_grad()`

**Performance Finding**: Training is still slow (~1.28hr/epoch vs ~2hr/epoch for full training) because gradients must still propagate through all 30 DiT blocks to reach img_emb:
```
loss → DiT blocks (30 layers) → context → clip_embedding → img_emb
              ↑
    Activation gradients still computed
```

**Future optimization options**:
1. Direct img_emb loss (skip DiT entirely)
2. Precompute latents/embeddings, train img_emb separately
3. Detach + straight-through estimator

## Inference Usage

```python
from examples.wanvideo.model_inference import Synthos_I2I as synthos

# Load model
pipe = synthos.load_synthos_i2i(
    checkpoint_path="./models/train/Synthos-I2I/epoch-10.safetensors",
    device="cuda",
)

# Generate with reference image (CLIP conditioning)
from PIL import Image
reference = Image.open("reference.jpg")

output = synthos.generate(
    pipe,
    prompt="an oil painting in impressionist style",
    reference_image=reference,
    height=480,
    width=832,
    cfg_scale=5.0,
    seed=42,
)
output.save("output.png")

# Generate without reference (pure T2I)
output = synthos.generate(
    pipe,
    prompt="a beautiful sunset",
    reference_image=None,
)
```

## Local Model Paths

- **DiT**: `/home/duality/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors`
- **T5, VAE, CLIP**: Downloaded from HuggingFace during training

## Bug Fixes

1. **Wandb I2I caption bug**: I2I validation images were showing T2I text prompts instead of input image paths. Fixed in `runner.py:78`.

2. **Validation uses empty prompt for I2I**: I2I validation now correctly uses `prompt=""` for pure CLIP conditioning.

## Git History

```
9f89505 Add train_img_emb_only mode for I2I training
26532cc Fix I2I validation to use empty prompt for pure CLIP conditioning
4c87308 added grad accum
c390e63 Fix validation scheduler state corruption by backing up/restoring sigmas array
49ace0a train works, but issue with val loop
20b908b Remove docs, update README with training/inference instructions
68f7444 Rename I2I model to Synthos-I2I
531c358 Add conditioning dropout hyperparameters to I2I training script
5c766ae Add validation during training with wandb logging
4ff704f Initial commit: Synthos - Video/Image generation framework
```

## Dataset Filtering

The full ContraStyles dataset has varied image sizes. To ensure quality:

1. **Filter by resolution**: `min_dim >= 512` keeps 134,848 images (37% of 363k)
2. **Train at 480x832**: Matches WAN pretraining resolution (~16:9)
3. **Center crop**: Images are resized and center-cropped to fit

```bash
python scripts/filter_metadata_by_resolution.py \
    --input data/contrastyles_full/metadata.csv \
    --output data/contrastyles_full/metadata_filtered.csv \
    --base_path data/contrastyles_full \
    --min_dim 512
```

Distribution of original dataset:
- min_dim >= 256: 279k (77%)
- min_dim >= 480: 151k (42%)
- min_dim >= 512: 135k (37%)
- min_dim >= 720: 67k (18%)

## Notes

- Training runs somewhere else (separate from this dev machine)
- Sample images for inference: `/home/duality/projects/DiffSynth-Studio/sample_images/`
- Checkpoints save to: `./models/train/Synthos-I2I/`
