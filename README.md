# Synthos

CLIP-conditioned image generation framework.

## Synthos-I2I

Image-to-image generation using CLIP cross-attention conditioning. The model generates images guided by:
- **CLIP embeddings** from a reference image (visual style/content)
- **Text prompts** (semantic guidance)

Generation starts from pure noise - the reference image influences output only through cross-attention, not latent initialization.

## Installation

```bash
git clone https://github.com/somepago/Synthos.git
cd Synthos
pip install -e .
pip install wandb
```

## Training

### 1. Prepare Dataset

```bash
# Download ContraStyles dataset (or use your own)
python scripts/download_contrastyles.py
```

Dataset structure:
```
data/contrastyles/
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── metadata.csv  # columns: video, prompt
```

### 2. Login to Services

```bash
wandb login
huggingface-cli login
```

### 3. Train

```bash
bash examples/wanvideo/model_training/full/Synthos-I2I.sh
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch_size` | Batch size per GPU | 1 |
| `--learning_rate` | Learning rate | 1e-5 |
| `--num_epochs` | Number of epochs | 10 |
| `--text_dropout_prob` | Text conditioning dropout (for CFG) | 0.1 |
| `--image_dropout_prob` | Image conditioning dropout (for CFG) | 0.1 |
| `--use_wandb` | Enable wandb logging | False |
| `--validate_steps` | Run validation every N steps | None |
| `--validation_prompts` | Validation prompts (pipe-separated) | None |
| `--validation_images` | Validation images (comma-separated) | None |

### Mixed Precision

Training uses bf16 by default via `accelerate launch --mixed_precision bf16`.

## Inference

### Python API

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
    width=480,
    cfg_scale=5.0,
    seed=42,
)
output.save("output.png")

# Generate without reference (pure T2I)
output = synthos.generate(
    pipe,
    prompt="a beautiful sunset",
    reference_image=None,
    height=480,
    width=480,
)
```

### CLI

```bash
python examples/wanvideo/model_inference/Synthos-I2I.py
```

## Architecture

```
Synthos-I2I
├── DiT backbone (from Wan2.1-T2V-1.3B)
├── T5 text encoder
├── CLIP image encoder (cross-attention conditioning)
└── VAE (output decoding only)
```

Key differences from standard I2V:
- No VAE latent initialization (`require_vae_embedding=False`)
- CLIP embedding injected via cross-attention only
- Trained with conditioning dropout for CFG support

## License

Apache 2.0




### ideas, useful
https://huggingface.co/datasets/laion/relaion-art/viewer/default/train?p=2&views%5B%5D=train
https://huggingface.co/datasets/laion/relaion-pop/viewer/default/train?p=2