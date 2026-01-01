# Synthos

Video and Image generation framework with I2I training support.

## Features

- **I2I Training**: Image-to-image training with CLIP conditioning
- **Wandb Integration**: Real-time training metrics and logging
- **Mixed Precision**: bf16/fp16 training for lower memory usage
- **Configurable Batch Size**: Adjust batch size for your GPU

## Setup

### 1. Create Environment

```bash
python -m venv ~/envs/synthos
source ~/envs/synthos/bin/activate
pip install -e .
pip install wandb
```

### 2. Login to Services

```bash
wandb login
huggingface-cli login
```

### 3. Download Dataset

```bash
python scripts/download_contrastyles.py  # Downloads 100 samples from ContraStyles
```

## Training

### I2I Training (Full Model)

```bash
source ~/envs/synthos/bin/activate
bash examples/wanvideo/model_training/full/Wan2.1-I2I-1.3B.sh
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch_size` | Batch size per GPU | 1 |
| `--use_wandb` | Enable wandb logging | False |
| `--wandb_project` | Wandb project name | diffsynth-training |
| `--wandb_run_name` | Custom run name | auto-generated |
| `--learning_rate` | Learning rate | 1e-5 |
| `--num_epochs` | Number of epochs | 10 |

### Mixed Precision

Add `--mixed_precision bf16` to `accelerate launch` for lower memory usage:

```bash
accelerate launch --mixed_precision bf16 examples/wanvideo/model_training/train.py ...
```

## Inference

### Text-to-Image

```python
import torch
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)

images = pipe(prompt="a beautiful sunset over mountains", num_frames=1)
images[0].save("output.png")
```

### Image-to-Image

```python
from PIL import Image

input_image = Image.open("input.jpg")
images = pipe(
    prompt="stylized painting of the scene",
    input_video=[input_image],
    denoising_strength=0.7,
    num_frames=1,
)
images[0].save("output_i2i.png")
```

## Project Structure

```
synthos/
├── diffsynth/
│   ├── pipelines/          # Inference pipelines
│   ├── diffusion/          # Training modules
│   │   ├── runner.py       # Training loop with wandb
│   │   └── parsers.py      # CLI arguments
│   └── models/             # Model definitions
├── examples/
│   └── wanvideo/
│       ├── model_inference/    # Inference scripts
│       └── model_training/     # Training scripts
└── data/                   # Training datasets
```

## License

Apache 2.0
