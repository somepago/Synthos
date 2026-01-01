import torch
from PIL import Image, ImageDraw
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# Load input image
input_image = Image.open("/home/duality/projects/DiffSynth-Studio/sample_images/1.jpg")

# Batched sampling across different denoising strengths
denoising_strengths = [0.0, 0.3, 0.5, 0.7, 1.0]
prompt = "a puppy is playing with a red ball"
seed = 42

results = []
for strength in denoising_strengths:
    print(f"Sampling with denoising_strength={strength}")
    images = pipe(
        prompt=prompt,
        negative_prompt="ugly, deformed, blurry, low quality",
        input_video=[input_image],
        denoising_strength=strength,
        height=input_image.height,
        width=input_image.width,
        num_frames=1,
        num_inference_steps=25,
        cfg_scale=5.0,
        seed=seed,
        tiled=True,
    )
    results.append((strength, images[0]))
    images[0].save(f"output_i2i_strength_{strength:.1f}_{prompt}.png")

# Create comparison grid
def create_grid(input_img, results, cols=5):
    """Create a grid with input image + all denoising strength results"""
    n = len(results) + 1  # +1 for input image
    rows = (n + cols - 1) // cols

    w, h = input_img.size
    label_height = 30
    grid = Image.new('RGB', (cols * w, rows * (h + label_height)), color='white')
    draw = ImageDraw.Draw(grid)

    # Add input image first
    grid.paste(input_img, (0, label_height))
    draw.text((5, 5), "Input", fill='black')

    # Add results
    for i, (strength, img) in enumerate(results):
        idx = i + 1
        row, col = idx // cols, idx % cols
        x, y = col * w, row * (h + label_height)
        grid.paste(img, (x, y + label_height))
        draw.text((x + 5, y + 5), f"strength={strength:.1f}", fill='black')

    return grid

grid = create_grid(input_image, results)
grid.save(f"output_i2i_comparison_{prompt}.png")
print(f"Saved comparison grid to output_i2i_comparison_{prompt}.png")
