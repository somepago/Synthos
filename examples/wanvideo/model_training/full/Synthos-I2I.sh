# Synthos-I2I: CLIP-conditioned Image Generation Training
#
# Architecture:
# - DiT backbone from Wan2.1-T2V-1.3B
# - CLIP image encoder for cross-attention conditioning
# - No VAE latent initialization (pure CLIP guidance)
#
# Training:
# - Text + image conditioning dropout for CFG
# - Starts from noise, CLIP embedding guides via cross-attention

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
  --batch_size 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Synthos-I2I" \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --i2i_mode \
  --text_dropout_prob 0.2 \
  --image_dropout_prob 0.2 \
  --use_wandb \
  --wandb_project "synthos-training" \
  --wandb_run_name "synthos-i2i-contrastyles" \
  --validate_steps 500 \
  --validation_prompts "stylized sunset over mountains|cyberpunk city at night|oil painting of a forest" \
  --validation_images "data/contrastyles/images/0001.jpg,data/contrastyles/images/0005.jpg,data/contrastyles/images/0010.jpg"
