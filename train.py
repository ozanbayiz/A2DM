import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
import torch.cuda.amp as amp  # For mixed precision training
import logging

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
        expanded_mask = mask.unsqueeze(-1).expand_as(data)
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
    kl_weight: float = 0.01,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    timer: Optional[Timer] = None,
    grad_clip: Optional[float] = None
) -> Dict[str, float]:
    """Train model for one epoch with mixed precision."""
    model.train()
    epoch_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
    num_batches = len(dataloader)
    
    # Update KL weight if supported
    if all(x is not None for x in [epoch, total_epochs]) and hasattr(model, 'update_kl_weight'):
        kl_weight = model.update_kl_weight(epoch, total_epochs)
        logger.info(f"KL weight updated to {COLORS.YELLOW}{kl_weight:.6f}{COLORS.RESET}")
    
    if timer:
        timer.lap()
    
    for batch_idx, (data, mask) in enumerate(dataloader):
        # Move data to device
        data = data.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True) if mask is not None else None
        
        # Mixed precision forward pass - handle both CUDA and MPS devices
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
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular optimization for CPU and MPS
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Update running losses
        epoch_losses['total'] += loss.item()
        epoch_losses['recon'] += recon_loss.item()
        epoch_losses['kl'] += kl_loss.item()
        
        # Log progress periodically
        if batch_idx % 10 == 0:
            logger.info(
                f"Batch [{batch_idx}/{num_batches}] - "
                f"Loss: {COLORS.MAGENTA}{loss.item():.4f}{COLORS.RESET} "
                f"(Recon: {COLORS.CYAN}{recon_loss.item():.4f}{COLORS.RESET}, "
                f"KL: {COLORS.GREEN}{kl_loss.item():.4f}{COLORS.RESET})"
            )
    
    # Calculate average losses
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    
    if timer:
        epoch_time = timer.lap()
        logger.info(
            f"Epoch time: {COLORS.CYAN}{Timer.format_time(epoch_time)}{COLORS.RESET} "
            f"(Total: {COLORS.CYAN}{Timer.format_time(timer.elapsed())}{COLORS.RESET})"
        )
    
    return avg_losses

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration."""
    logger = setup_logger()
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # Setup device and mixed precision training
    device = torch.device(cfg.training.device)
    # Only enable GradScaler for CUDA devices
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    logger.info(f"Using device: {COLORS.BOLD}{COLORS.GREEN}{device}{COLORS.RESET}")
    
    # Set up directories with better organization
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
        if device.type == 'cuda':
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        
        # Initialize dataset and dataloader
        dataset = get_dataset(name=cfg.dataset.name, **cfg.dataset.params)
        train_loader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
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
        timer = Timer().start()
        
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
                grad_clip=cfg.training.get('gradient_clipping')
            )
            
            # Update history and save checkpoints
            for key in train_losses:
                losses[f'train_{key}'].append(train_losses[key])
            
            # Save best model
            if train_losses['total'] < best_loss:
                best_loss = train_losses['total']
                save_checkpoint(
                    model=model,
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
                    model=model,
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
                    num_params=num_params,
                    logger=logger
                )
        
        # Save final checkpoint and plots
        save_checkpoint(
            model=model,
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
        logger.error(f"Training failed: {COLORS.RED}{str(e)}{COLORS.RESET}")
        raise

if __name__ == "__main__":
    main()