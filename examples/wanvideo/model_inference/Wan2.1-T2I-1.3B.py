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
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# Text-to-Image (single frame generation)
images = pipe(
    prompt="a beautiful sunset over mountains, golden hour lighting, dramatic clouds",
    negative_prompt="ugly, deformed, blurry, low quality",
    height=480,
    width=832,
    num_frames=1,  # Single frame = image
    num_inference_steps=50,
    cfg_scale=5.0,
    seed=42,
    tiled=True,
)

# Save the generated image
images[0].save("output_t2i.png")
print(f"Saved image to output_t2i.png")
