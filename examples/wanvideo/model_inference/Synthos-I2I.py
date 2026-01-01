"""
Synthos-I2I: CLIP-conditioned Image Generation

This model uses CLIP cross-attention conditioning only (no VAE latent initialization).
Generation starts from pure noise and is guided by:
- CLIP image embeddings (visual style/content from reference image)
- Text prompts (semantic guidance)

The reference image influences generation purely through cross-attention,
not through latent space initialization.
"""

import torch
from PIL import Image, ImageDraw
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict


def load_synthos_i2i(
    checkpoint_path=None,
    device="cuda",
    torch_dtype=torch.bfloat16,
):
    """
    Load Synthos-I2I model.

    Args:
        checkpoint_path: Path to trained checkpoint. If None, uses base model.
        device: Device to load model on.
        torch_dtype: Model precision.

    Returns:
        WanVideoPipeline configured for CLIP-conditioned generation.
    """
    # Base models + CLIP encoder
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            # DiT backbone
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            # Text encoder
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            # VAE (for decoding output only)
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
            # CLIP image encoder for conditioning
            ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )

    # Load trained checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = load_state_dict(checkpoint_path)
        pipe.dit.load_state_dict(state_dict, strict=False)
        print(f"Loaded {len(state_dict)} parameters from checkpoint")

    # Enable CLIP-only mode (no VAE latent initialization)
    pipe.dit.has_clip_input = True
    pipe.dit.require_vae_embedding = False

    return pipe


def generate(
    pipe,
    prompt,
    reference_image=None,
    negative_prompt="ugly, deformed, blurry, low quality",
    height=480,
    width=480,
    num_inference_steps=25,
    cfg_scale=5.0,
    seed=42,
):
    """
    Generate image with optional CLIP reference conditioning.

    Args:
        pipe: Loaded Synthos-I2I pipeline.
        prompt: Text prompt for generation.
        reference_image: Optional PIL Image for CLIP conditioning.
                        If None, generates from text only (T2I mode).
        negative_prompt: Negative prompt.
        height: Output height.
        width: Output width.
        num_inference_steps: Number of diffusion steps.
        cfg_scale: Classifier-free guidance scale.
        seed: Random seed.

    Returns:
        Generated PIL Image.
    """
    if isinstance(reference_image, str):
        reference_image = Image.open(reference_image)

    # Build generation kwargs
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": 1,
        "num_inference_steps": num_inference_steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "tiled": True,
    }

    # Add reference image for CLIP conditioning if provided
    if reference_image is not None:
        kwargs["input_image"] = reference_image

    images = pipe(**kwargs)
    return images[0]


if __name__ == "__main__":
    # Load model
    pipe = load_synthos_i2i(
        checkpoint_path=None,  # e.g., "./models/train/Synthos-I2I/epoch-10.safetensors"
        device="cuda",
    )

    # Load reference image for CLIP conditioning
    reference_image = Image.open("/home/duality/projects/DiffSynth-Studio/sample_images/1.jpg")

    # Test prompts with same reference image
    prompts = [
        "a photorealistic scene",
        "an oil painting in impressionist style",
        "a cyberpunk digital art",
        "a watercolor sketch",
    ]

    results = []
    for prompt in prompts:
        print(f"Generating: {prompt}")
        output = generate(
            pipe,
            prompt=prompt,
            reference_image=reference_image,
            height=480,
            width=480,
            cfg_scale=5.0,
            seed=42,
        )
        results.append((prompt, output))
        # Save with sanitized filename
        filename = prompt.replace(" ", "_")[:30]
        output.save(f"synthos_{filename}.png")

    # Also generate without reference (pure T2I)
    print("Generating T2I (no reference)...")
    t2i_output = generate(
        pipe,
        prompt="a beautiful sunset over mountains",
        reference_image=None,
        height=480,
        width=480,
        cfg_scale=5.0,
        seed=42,
    )
    t2i_output.save("synthos_t2i_sunset.png")

    # Create comparison grid
    def create_grid(ref_img, results, cols=3):
        n = len(results) + 1
        rows = (n + cols - 1) // cols
        w, h = 480, 480
        label_height = 40
        grid = Image.new('RGB', (cols * w, rows * (h + label_height)), color='white')
        draw = ImageDraw.Draw(grid)

        # Resize reference image
        ref_resized = ref_img.resize((w, h))
        grid.paste(ref_resized, (0, label_height))
        draw.text((5, 5), "Reference (CLIP)", fill='black')

        for i, (prompt, img) in enumerate(results):
            idx = i + 1
            row, col = idx // cols, idx % cols
            x, y = col * w, row * (h + label_height)
            img_resized = img.resize((w, h))
            grid.paste(img_resized, (x, y + label_height))
            draw.text((x + 5, y + 5), prompt[:25], fill='black')

        return grid

    grid = create_grid(reference_image, results)
    grid.save("synthos_comparison.png")
    print("Saved comparison grid to synthos_comparison.png")
