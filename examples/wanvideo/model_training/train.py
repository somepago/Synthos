import torch, os, argparse, accelerate, warnings, random
import torch.nn as nn
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import MLP
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        # I2I variation mode: CLIP conditioning without VAE embedding
        i2i_mode=False,
        # Train only img_emb layer (freeze rest of DiT)
        train_img_emb_only=False,
        # Conditioning dropout for classifier-free guidance style training
        text_dropout_prob=0.1,
        image_dropout_prob=0.1,
    ):
        super().__init__()
        self.text_dropout_prob = text_dropout_prob
        self.image_dropout_prob = image_dropout_prob
        self.train_img_emb_only = train_img_emb_only

        # Warning - but allow disabling for train_img_emb_only mode
        if not use_gradient_checkpointing and not train_img_emb_only:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Option 1: Disable gradient checkpointing for train_img_emb_only (faster, uses more VRAM)
        if train_img_emb_only:
            use_gradient_checkpointing = False
            print("train_img_emb_only: Disabled gradient checkpointing for faster training")

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)

        # I2I mode: Add CLIP embedding layer to T2V model
        self.i2i_mode = i2i_mode
        if i2i_mode:
            self._setup_i2i_mode()

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # If train_img_emb_only, freeze everything except img_emb
        if train_img_emb_only and i2i_mode:
            self._freeze_except_img_emb()

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def _setup_i2i_mode(self):
        """
        Setup I2I variation mode:
        - Add has_clip_input flag to DiT
        - Initialize img_emb layer (CLIP projection) if not present
        - Disable VAE embedding (we only want CLIP conditioning)
        - This allows CLIP-based image conditioning without VAE embedding
        """
        dit = self.pipe.dit

        # Enable CLIP input mode
        dit.has_clip_input = True

        # Disable VAE embedding - we only want CLIP conditioning for variations
        dit.require_vae_embedding = False

        # Add img_emb layer if not present (T2V models don't have it)
        if not hasattr(dit, 'img_emb') or dit.img_emb is None:
            print("I2I mode: Adding img_emb layer for CLIP conditioning")
            dit.img_emb = MLP(1280, dit.dim, has_pos_emb=False)
            # Initialize with small weights for stable training start
            for module in dit.img_emb.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            dit.img_emb = dit.img_emb.to(device=next(dit.parameters()).device, dtype=next(dit.parameters()).dtype)

        # Ensure image encoder is available
        if self.pipe.image_encoder is None:
            raise ValueError("I2I mode requires image_encoder. Add CLIP model to model_configs.")

        print(f"I2I mode enabled: has_clip_input={dit.has_clip_input}, require_vae_embedding={dit.require_vae_embedding}")

    def _freeze_except_img_emb(self):
        """Freeze all DiT parameters except img_emb layer."""
        dit = self.pipe.dit

        # First freeze everything in DiT
        for param in dit.parameters():
            param.requires_grad = False

        # Then unfreeze only img_emb
        if hasattr(dit, 'img_emb') and dit.img_emb is not None:
            for param in dit.img_emb.parameters():
                param.requires_grad = True

            # Count trainable params
            trainable_params = sum(p.numel() for p in dit.img_emb.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in dit.parameters())
            print(f"train_img_emb_only: Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        else:
            raise ValueError("train_img_emb_only requires img_emb layer. Enable --i2i_mode first.")

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        # Apply conditioning dropout for classifier-free guidance training
        prompt = data["prompt"]
        drop_text = random.random() < self.text_dropout_prob
        drop_image = random.random() < self.image_dropout_prob

        if drop_text:
            prompt = ""  # Empty prompt for unconditional

        inputs_posi = {"prompt": prompt}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "_drop_image": drop_image,  # Flag for image dropout
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)

        # Apply image dropout
        if drop_image and "input_image" in inputs_shared:
            inputs_shared["input_image"] = None

        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)

        # Option 2: Run frozen encoder units in no_grad for speed
        if self.train_img_emb_only:
            # Run units (text/CLIP encoders) without gradient tracking
            # This is safe because:
            # - Text encoder is frozen → no gradients needed
            # - CLIP encoder is frozen → no gradients needed
            # - clip_feature will be used by img_emb which HAS gradients
            with torch.no_grad():
                for unit in self.pipe.units:
                    inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        else:
            for unit in self.pipe.units:
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    # I2I variation mode
    parser.add_argument("--i2i_mode", default=False, action="store_true", help="Enable I2I variation mode: CLIP-only image conditioning without VAE embedding.")
    parser.add_argument("--train_img_emb_only", default=False, action="store_true", help="Train only img_emb layer (freeze rest of DiT). Requires --i2i_mode.")
    # Conditioning dropout for CFG training
    parser.add_argument("--text_dropout_prob", type=float, default=0.1, help="Probability of dropping text conditioning (for CFG training).")
    parser.add_argument("--image_dropout_prob", type=float, default=0.1, help="Probability of dropping image conditioning (for CFG training).")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        i2i_mode=args.i2i_mode,
        train_img_emb_only=args.train_img_emb_only,
        text_dropout_prob=args.text_dropout_prob,
        image_dropout_prob=args.image_dropout_prob,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
