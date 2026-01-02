# Synthos-I2I Phase 1: Train img_emb layer only
#
# This script trains ONLY the CLIP projection layer (img_emb) while freezing
# all other DiT parameters. This teaches the model to use CLIP embeddings
# before fine-tuning the full model.
#
# After this phase, you can:
# 1. Continue with full DiT fine-tuning using the checkpoint from this phase
# 2. Or use LoRA on attention layers

accelerate launch --mixed_precision bf16 examples/wanvideo/model_training/train.py \
  --dataset_base_path data/contrastyles_full \
  --dataset_metadata_path data/contrastyles_full/metadata.csv \
  --height 480 \
  --width 480 \
  --num_frames 1 \
  --dataset_repeat 1 \
  --model_paths '["/home/duality/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"]' \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-1.3B-InP:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --batch_size 32 \
  --gradient_accumulation_steps 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Synthos-I2I-img-emb-only" \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --i2i_mode \
  --train_img_emb_only \
  --text_dropout_prob 0.9 \
  --image_dropout_prob 0.0 \
  --use_wandb \
  --wandb_project "synthos-training" \
  --wandb_run_name "synthos-i2i-img-emb-only-full" \
  --validate_steps 100 \
  --validation_prompts "stylized sunset over mountains|cyberpunk city at night|oil painting of a forest" \
  --validation_images "data/contrastyles_full/images/00001/000010007.jpg,data/contrastyles_full/images/00001/000010000.jpg,data/contrastyles_full/images/00002/000020000.jpg"
