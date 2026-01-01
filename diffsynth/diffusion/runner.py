import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from PIL import Image

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def run_validation(model, prompts, height, width, num_frames, output_dir, global_step, use_wandb, input_images=None):
    """Generate validation samples during training.

    Note: Validation is run in a separate process or with careful state management
    to avoid corrupting the training pipeline state.
    """
    import copy

    os.makedirs(output_dir, exist_ok=True)
    images_to_log = []

    # Save scheduler state before validation
    pipe = model.module.pipe if hasattr(model, 'module') else model.pipe
    scheduler_timesteps_backup = pipe.scheduler.timesteps.clone() if hasattr(pipe.scheduler, 'timesteps') else None
    scheduler_sigmas_backup = pipe.scheduler.sigmas.clone() if hasattr(pipe.scheduler, 'sigmas') else None
    scheduler_training_backup = getattr(pipe.scheduler, 'training', True)

    try:
        model.eval()

        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                try:
                    # T2I generation (text-to-image only for validation)
                    result = pipe(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames if num_frames else 1,
                        num_inference_steps=20,
                        cfg_scale=5.0,
                        seed=42,
                        tiled=True,
                    )

                    if result and len(result) > 0:
                        img = result[0]
                        save_path = os.path.join(output_dir, f"step_{global_step}_t2i_{i}.png")
                        img.save(save_path)
                        images_to_log.append(("t2i", prompt, img))

                    # I2I generation with CLIP conditioning (no VAE init)
                    if input_images and i < len(input_images):
                        input_img = input_images[i]
                        if isinstance(input_img, str):
                            input_img = Image.open(input_img)

                        result = pipe(
                            prompt=prompt,
                            input_image=input_img,  # CLIP conditioning only
                            height=height,
                            width=width,
                            num_frames=1,
                            num_inference_steps=20,
                            cfg_scale=5.0,
                            seed=42,
                            tiled=True,
                        )

                        if result and len(result) > 0:
                            img = result[0]
                            save_path = os.path.join(output_dir, f"step_{global_step}_i2i_{i}.png")
                            img.save(save_path)
                            images_to_log.append(("i2i", prompt, img))

                except Exception as e:
                    print(f"Validation error for prompt '{prompt}': {e}")
                    import traceback
                    traceback.print_exc()

        # Log to wandb
        if use_wandb and WANDB_AVAILABLE and images_to_log:
            wandb_images = []
            for img_type, prompt, img in images_to_log:
                wandb_images.append(wandb.Image(img, caption=f"{img_type}: {prompt[:50]}"))
            wandb.log({"validation/samples": wandb_images}, step=global_step)

    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore model to training mode
        model.train()

        # Restore scheduler state
        if scheduler_timesteps_backup is not None:
            pipe.scheduler.timesteps = scheduler_timesteps_backup
        if scheduler_sigmas_backup is not None:
            pipe.scheduler.sigmas = scheduler_sigmas_backup
        # Restore scheduler.training flag (critical for input_latents to be returned)
        pipe.scheduler.training = scheduler_training_backup

    return len(images_to_log)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    batch_size: int = 1,
    use_wandb: bool = False,
    wandb_project: str = "diffsynth-training",
    wandb_run_name: str = None,
    validate_steps: int = None,
    validation_prompts: list = None,
    validation_images: list = None,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        batch_size = getattr(args, 'batch_size', 1)
        use_wandb = getattr(args, 'use_wandb', False)
        wandb_project = getattr(args, 'wandb_project', 'diffsynth-training')
        wandb_run_name = getattr(args, 'wandb_run_name', None)
        validate_steps = getattr(args, 'validate_steps', None)
        validation_prompts = getattr(args, 'validation_prompts', None)
        validation_images = getattr(args, 'validation_images', None)

    # Parse validation prompts from comma-separated string
    if isinstance(validation_prompts, str):
        validation_prompts = [p.strip() for p in validation_prompts.split('|')]
    if isinstance(validation_images, str):
        validation_images = [os.path.abspath(p.strip()) for p in validation_images.split(',')]

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE and accelerator.is_main_process:
        wandb_config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": accelerator.gradient_accumulation_steps,
            "dataset_size": len(dataset),
        }
        if args is not None:
            wandb_config.update({
                "height": getattr(args, 'height', None),
                "width": getattr(args, 'width', None),
                "num_frames": getattr(args, 'num_frames', None),
                "lora_rank": getattr(args, 'lora_rank', None),
                "task": getattr(args, 'task', None),
            })
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=wandb_config,
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not installed. Run `pip install wandb`")

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # Support batch_size > 1
    if batch_size > 1:
        def collate_batch(batch):
            return batch  # Return list of samples for batch processing
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_batch, num_workers=num_workers
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers
        )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    global_step = 0
    for epoch_id in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for data in tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}"):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # Handle batch processing
                if batch_size > 1 and isinstance(data, list):
                    batch_loss = 0.0
                    for sample in data:
                        if dataset.load_from_cache:
                            loss = model({}, inputs=sample)
                        else:
                            loss = model(sample)
                        batch_loss += loss
                    loss = batch_loss / len(data)
                else:
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)

                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

                # Logging
                global_step += 1
                loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
                epoch_loss += loss_value
                num_batches += 1

                if use_wandb and WANDB_AVAILABLE and accelerator.is_main_process:
                    wandb.log({
                        "train/loss": loss_value,
                        "train/epoch": epoch_id,
                        "train/global_step": global_step,
                        "train/lr": scheduler.get_last_lr()[0],
                    }, step=global_step)

                # Run validation
                if (validate_steps and validation_prompts and
                    global_step % validate_steps == 0 and accelerator.is_main_process):
                    print(f"\nRunning validation at step {global_step}...")
                    val_output_dir = os.path.join(model_logger.output_path, "validation")
                    height = getattr(args, 'height', 480) if args else 480
                    width = getattr(args, 'width', 480) if args else 480
                    num_frames = getattr(args, 'num_frames', 1) if args else 1
                    num_samples = run_validation(
                        model, validation_prompts, height, width, num_frames,
                        val_output_dir, global_step, use_wandb, validation_images
                    )
                    print(f"Generated {num_samples} validation samples")

        # Log epoch metrics
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        if use_wandb and WANDB_AVAILABLE and accelerator.is_main_process:
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch": epoch_id + 1,
            }, step=global_step)

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)

    if use_wandb and WANDB_AVAILABLE and accelerator.is_main_process:
        wandb.finish()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
