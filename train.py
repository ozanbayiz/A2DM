import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
import torch.cuda.amp as amp  # For mixed precision training
import logging
import os

from utils.training import (
    setup_logger,
    save_checkpoint,
    plot_training_curves,
    Timer,
    COLORS,
)
from utils.factory import get_model, get_dataset

def compute_loss(
    model: torch.nn.Module,
    data: torch.Tensor, 
    recon: torch.Tensor,  
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: Optional[torch.Tensor] = None, 
    kl_weight: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss with optional masking."""
    # Apply mask if provided
    if mask is not None:
        # Expand mask to match data dimensions
        expanded_mask = mask.unsqueeze(2).expand_as(data)
        valid_data = data * expanded_mask
        valid_recon = recon * expanded_mask
    else:
        valid_data = data
        valid_recon = recon
    
    # Use model's custom loss if available
    if hasattr(model, 'compute_loss'):
        return model.compute_loss(valid_data, valid_recon, mu, logvar, kl_weight_override=kl_weight)
    
    # Default loss calculation (vectorized operations)
    recon_loss = torch.nn.functional.mse_loss(valid_recon, valid_data)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    logger: logging.Logger,
    kl_weight: float = 1,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    timer: Optional[Timer] = None,
    grad_clip: Optional[float] = None,
    distributed: bool = False,
    rank: int = 0,
) -> Dict[str, float]:
    """Train model for one epoch with mixed precision and distributed support."""
    model.train()
    epoch_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
    num_batches = len(dataloader)
    
    # Update KL weight if supported
    if all(x is not None for x in [epoch, total_epochs]) and hasattr(model, 'update_kl_weight'):
        kl_weight = model.update_kl_weight(epoch, total_epochs)
        if not distributed or rank == 0:
            logger.info(f"KL weight updated to {COLORS.YELLOW}{kl_weight:.6f}{COLORS.RESET}")
    
    if timer and (not distributed or rank == 0):
        timer.lap()
    
    # Update distributed sampler with epoch for proper shuffling
    if distributed and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)
    
    for batch_idx, (data, mask) in enumerate(dataloader):
        # Move data to device
        data = data.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True) if mask is not None else None
        
        # Mixed precision forward pass
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                recon, mu, logvar = model(data)
                loss, recon_loss, kl_loss = compute_loss(
                    model, data, recon, mu, logvar, mask, kl_weight
                )
        else:
            # For CPU and MPS, run without autocast
            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = compute_loss(
                model, data, recon, mu, logvar, mask, kl_weight
            )
        
        # Optimize with gradient scaling only for CUDA
        optimizer.zero_grad(set_to_none=True)
        
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Update running losses
        epoch_losses['total'] += loss.item()
        epoch_losses['recon'] += recon_loss.item()
        epoch_losses['kl'] += kl_loss.item()
        
        # Log progress periodically (only from main process)
        if (batch_idx % 10 == 0) and (not distributed or rank == 0):
            logger.info(
                f"Batch [{batch_idx}/{num_batches}] - "
                f"Loss: {COLORS.MAGENTA}{loss.item():.4f}{COLORS.RESET} "
                f"(Recon: {COLORS.CYAN}{recon_loss.item():.4f}{COLORS.RESET}, "
                f"KL: {COLORS.GREEN}{kl_loss.item():.4f}{COLORS.RESET})"
            )
    
    # For distributed training, synchronize losses across processes
    if distributed:
        for k in epoch_losses:
            tensor = torch.tensor(epoch_losses[k], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            epoch_losses[k] = tensor.item() / dist.get_world_size()
    
    # Calculate average losses
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    
    if timer and (not distributed or rank == 0):
        epoch_time = timer.lap()
        logger.info(
            f"Epoch time: {COLORS.CYAN}{Timer.format_time(epoch_time)}{COLORS.RESET} "
            f"(Total: {COLORS.CYAN}{Timer.format_time(timer.elapsed())}{COLORS.RESET})"
        )
    
    return avg_losses

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with distributed training support."""
    
    # Initialize distributed environment if enabled
    distributed = cfg.training.get('distributed', False)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if distributed:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            # Initialize process group
            dist.init_process_group(backend="nccl")
        else:
            device = torch.device("cpu")
            # Use gloo backend for CPU
            dist.init_process_group(backend="gloo")
        
        # Set device for this process
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device(cfg.training.device)
    
    # Only rank 0 should print logs and save files
    if not distributed or rank == 0:
        logger = setup_logger()
    else:
        # Create a dummy logger for non-rank-0 processes
        logger = logging.getLogger("dummy_logger")
        logger.addHandler(logging.NullHandler())
    
    if rank == 0 or not distributed:
        logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
        if distributed:
            logger.info(f"Distributed training enabled: {world_size} processes")
            logger.info(f"Process rank: {rank}, local rank: {local_rank}, world size: {world_size}")
    
    # Setup mixed precision training
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    if rank == 0 or not distributed:
        logger.info(f"Using device: {COLORS.BOLD}{COLORS.GREEN}{device}{COLORS.RESET}")
    
    # Set up directories with better organization (only on rank 0)
    if rank == 0 or not distributed:
        work_dir = Path(hydra.utils.get_original_cwd())
        model_dir = work_dir / "outputs" / f"{cfg.model.name}_{cfg.dataset.name}"
        checkpoints_dir = model_dir / "checkpoints"
        plots_dir = model_dir / "plots"
        config_dir = model_dir / "config"
        
        # Create all directories
        for dir_path in [checkpoints_dir, plots_dir, config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save full configuration
        config_path = config_dir / "config.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(cfg, f)
        logger.info(f"Saved configuration to {COLORS.GREEN}{config_path}{COLORS.RESET}")
    
    try:
        # Initialize model and move to device
        model = get_model(name=cfg.model.name, config=cfg.model.params)
        model = model.to(device)
        
        # Wrap model in DDP for distributed training
        if distributed:
            model = DDP(model, device_ids=[local_rank] if device.type != "cpu" else None)
            
        # Initialize dataset
        dataset = get_dataset(name=cfg.dataset.name, **cfg.dataset.params)
        
        # Use DistributedSampler for distributed training
        if distributed:
            train_sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False
            )
            shuffle = False  # Sampler handles shuffling
        else:
            train_sampler = None
            shuffle = True
        
        # Initialize dataloader
        train_loader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=cfg.training.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True if cfg.training.get('num_workers', 4) > 0 else False
        )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            eps=1e-8,
        )
        
        # Training state
        num_params = sum(p.numel() for p in model.parameters())
        best_loss = float('inf')
        losses = {k: [] for k in ['train_total', 'train_recon', 'train_kl']}
        timer = Timer().start() if not distributed or rank == 0 else None
        
        if rank == 0 or not distributed:
            logger.info(f"Starting training: {num_params:,} parameters, {len(train_loader)} batches per epoch")
        
        # Training loop
        for epoch in range(cfg.training.epochs):
            train_losses = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                logger=logger,
                kl_weight=cfg.training.get('kl_weight', 0.01),
                epoch=epoch,
                total_epochs=cfg.training.epochs,
                timer=timer,
                grad_clip=cfg.training.get('gradient_clipping', None),
                distributed=distributed,
                rank=rank
            )
            
            # Update history and save checkpoints (only on rank 0)
            if not distributed or rank == 0:
                # Update history
                for key in train_losses:
                    losses[f'train_{key}'].append(train_losses[key])
                
                # Save best model
                if train_losses['total'] < best_loss:
                    best_loss = train_losses['total']
                    save_checkpoint(
                        model=model.module if distributed else model,
                        optimizer=optimizer,
                        epoch=epoch,
                        losses=losses,
                        config=cfg,
                        filepath=checkpoints_dir / "best_model.pt",
                        logger=logger
                    )
                
                # Save periodic checkpoints and plots
                if (epoch + 1) % cfg.training.get('save_interval', 10) == 0:
                    # Save checkpoint
                    save_checkpoint(
                        model=model.module if distributed else model,
                        optimizer=optimizer,
                        epoch=epoch,
                        losses=losses,
                        config=cfg,
                        filepath=checkpoints_dir / f"checkpoint_epoch_{epoch+1}.pt",
                        logger=logger
                    )
                    
                    # Save training curves
                    plot_training_curves(
                        losses,
                        output_path=plots_dir / f"training_curves_epoch_{epoch+1}.png",
                        model_name=cfg.model.name,
                        dataset_name=cfg.dataset.name,
                        num_params=num_params,
                        logger=logger
                    )
        
        # Save final checkpoint and plots (only on rank 0)
        if not distributed or rank == 0:
            save_checkpoint(
                model=model.module if distributed else model,
                optimizer=optimizer,
                epoch=cfg.training.epochs-1,
                losses=losses,
                config=cfg,
                filepath=checkpoints_dir / "final_model.pt",
                logger=logger
            )
            
            plot_training_curves(
                losses,
                output_path=plots_dir / "final_training_curves.png",
                model_name=cfg.model.name,
                num_params=num_params,
                logger=logger
            )
            
            logger.info(f"{COLORS.BOLD}Training Complete - Best Loss: {COLORS.MAGENTA}{best_loss:.4f}{COLORS.RESET}")
    
    except Exception as e:
        if rank == 0 or not distributed:
            logger.error(f"Training failed: {COLORS.RED}{str(e)}{COLORS.RESET}")
        raise
    
    finally:
        # Clean up distributed processing group
        if distributed:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()