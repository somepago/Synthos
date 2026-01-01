# I2I Variation Model Training
# Base: Wan2.1-T2V-1.3B + CLIP image encoder (from I2V model)
# Trains img_emb layer (CLIP projection) + optionally finetunes DiT
#
# Architecture:
# - DiT from T2V-1.3B (in_dim=16, no VAE embedding)
# - CLIP encoder from I2V model (ViT-H, 257 tokens per image)
# - New img_emb layer initialized and trained
#
# Training data: Single images with prompts
# The model learns to generate variations guided by CLIP embedding

# Mixed precision: bf16 for lower memory, fp16 also available
accelerate launch --mixed_precision bf16 examples/wanvideo/model_training/train.py \
  --dataset_base_path data/contrastyles \
  --dataset_metadata_path data/contrastyles/metadata.csv \
  --height 480 \
  --width 480 \
  --num_frames 1 \
  --dataset_repeat 100 \
  --model_paths '["/home/duality/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"]' \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-1.3B-InP:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --batch_size 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-I2I-1.3B_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --i2i_mode \
  --use_wandb \
  --wandb_project "wan-i2i-training" \
  --wandb_run_name "i2i-contrastyles-full"
