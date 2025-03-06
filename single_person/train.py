import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

# Import necessary models
import sys
import os
from utils import combined_motion_loss

# Get absolute path to the project root and add it to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

try:
    from models.convnext.vae import ConvNextVAE
    from models.transformer.vae import TransformerVAE
    print("Successfully imported ConvNextVAE")
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Looking for file at: {os.path.join(project_root, 'models', 'convnext', 'convnext_vae2.py')}")
    print(f"File exists: {os.path.exists(os.path.join(project_root, 'models', 'convnext', 'convnext_vae2.py'))}")
    raise

# Import dataloaders
from single_person.dataloader import (
    BobTranslationDataset,
    BobOrientationDataset,
    BobPoseDataset,
)

# ANSI color codes for colored terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

# Custom formatter for colored logging
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Format timestamp with color
        timestamp = self.formatTime(record, self.datefmt)
        timestamp = f"{Colors.CYAN}{timestamp}{Colors.RESET}"
        
        # Format log level with color
        levelname = record.levelname
        if levelname == 'INFO':
            levelname = f"{Colors.GREEN}{levelname}{Colors.RESET}"
        elif levelname == 'WARNING':
            levelname = f"{Colors.YELLOW}{levelname}{Colors.RESET}"
        elif levelname == 'ERROR':
            levelname = f"{Colors.RED}{levelname}{Colors.RESET}"
        elif levelname == 'CRITICAL':
            levelname = f"{Colors.BG_RED}{Colors.WHITE}{levelname}{Colors.RESET}"
        
        # Format logger name with color
        name = f"{Colors.BLUE}{record.name}{Colors.RESET}"
        
        # Format the message
        msg = record.getMessage()
        
        # Handle specific keywords in the message with colors
        if "Epoch" in msg:
            msg = msg.replace("Epoch", f"{Colors.YELLOW}Epoch{Colors.RESET}")
        if "Loss:" in msg:
            msg = msg.replace("Loss:", f"{Colors.BOLD}{Colors.MAGENTA}Loss:{Colors.RESET}")
        if "(Recon:" in msg:
            msg = msg.replace("(Recon:", f"({Colors.CYAN}Recon:{Colors.RESET}")
        if "KL:" in msg:
            msg = msg.replace("KL:", f"{Colors.GREEN}KL:{Colors.RESET}")
        if "LR:" in msg:
            msg = msg.replace("LR:", f"{Colors.BOLD}{Colors.BLUE}LR:{Colors.RESET}")
        if "Model:" in msg:
            msg = msg.replace("Model:", f"{Colors.BOLD}{Colors.WHITE}Model:{Colors.RESET}")
            
        # Build the colored log message
        log_msg = f"{timestamp} - {name} - {levelname} - {msg}"
        return log_msg

# Set up logging with colors
logger = logging.getLogger('vae_training_single_person')
logger.setLevel(logging.INFO)

# Remove existing handlers
for handler in logger.handlers:
    logger.removeHandler(handler)

# Create console handler with colored formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(console_handler)

# Create a file handler for permanent log storage
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / f"single_person_vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler.setFormatter(logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(file_handler)

# Timer class for training time tracking
class Timer:
    def __init__(self):
        self.start_time = None
        self.lap_time = None
    
    def start(self):
        self.start_time = time.time()
        self.lap_time = self.start_time
        return self
    
    def lap(self):
        current_time = time.time()
        elapsed = current_time - self.lap_time
        self.lap_time = current_time
        return elapsed
    
    def elapsed(self):
        return time.time() - self.start_time
    
    @staticmethod
    def format_time(seconds):
        return str(timedelta(seconds=int(seconds)))

def compute_loss(model, data, recon, mu, logvar, mask=None, kl_weight=0.01, velocity_weight=0.5):
    """Unified loss computation function supporting masked data and model-specific loss"""
    
    # Apply mask if available
    if mask is not None:
        valid_data = data * mask[:, :, None]
        valid_recon = recon * mask[:, :, None]
    else:
        valid_data = data
        valid_recon = recon
        
    # Use model's compute_loss if available
    if hasattr(model, 'compute_loss'):
        loss, recon_loss, kl_loss = model.compute_loss(valid_data, valid_recon, mu, logvar, kl_weight_override=kl_weight)
    else:
        # Fallback to basic loss calculation
        # Use combined_motion_loss from utils.py
        loss_dict = combined_motion_loss(valid_data, valid_recon)
        recon_loss = loss_dict['total']
        geo_loss = loss_dict['geodesic']
        vel_loss = loss_dict['velocity']
        accel_loss = loss_dict['acceleration']
            
        # KL divergence remains the same
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Combined loss
        loss = recon_loss + kl_loss * kl_weight
    
    return loss, recon_loss, kl_loss

def train_epoch(model, dataloader, optimizer, device, kl_weight=0.01, epoch=None, total_epochs=None, timer=None):
    """Train for one epoch with timing"""
    model.train()
    epoch_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
    num_batches = len(dataloader)
    
    # Update KL weight if model supports it
    if epoch is not None and hasattr(model, 'update_kl_weight') and total_epochs is not None:
        kl_weight = model.update_kl_weight(epoch, total_epochs)
        logger.info(f"KL weight updated to {Colors.YELLOW}{kl_weight:.6f}{Colors.RESET}")
    
    # Start timing this epoch
    if timer:
        epoch_start = timer.lap()
    
    progress_bar = tqdm(enumerate(dataloader), total=num_batches, 
                        desc=f"Epoch {Colors.YELLOW}{epoch+1}/{total_epochs}{Colors.RESET}")
    
    for i, (data, mask) in progress_bar:
        data = data.to(device)
        mask = mask.to(device) if mask is not None else None
        
        # Forward pass
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        
        # Compute loss
        loss, recon_loss, kl_loss = compute_loss(model, data, recon, mu, logvar, mask, kl_weight)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update running losses
        epoch_losses['total'] += loss.item()
        epoch_losses['recon'] += recon_loss.item()
        epoch_losses['kl'] += kl_loss.item()
        
        # Update progress bar
        if i % 5 == 0:
            progress_bar.set_postfix({
                'loss': f"{Colors.MAGENTA}{loss.item():.4f}{Colors.RESET}",
                'recon': f"{Colors.CYAN}{recon_loss.item():.4f}{Colors.RESET}", 
                'kl': f"{Colors.GREEN}{kl_loss.item():.4f}{Colors.RESET}"
            })
    
    # Calculate average losses
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    
    # Log epoch time
    if timer:
        epoch_time = timer.lap()
        logger.info(f"Epoch time: {Colors.CYAN}{Timer.format_time(epoch_time)}{Colors.RESET} " +
                   f"(Total: {Colors.CYAN}{Timer.format_time(timer.elapsed())}{Colors.RESET})")
    
    return avg_losses

def validate(model, dataloader, device, kl_weight=0.01):
    """Validate the model on a separate dataset"""
    model.eval()
    val_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for data, mask in dataloader:
            data = data.to(device)
            mask = mask.to(device) if mask is not None else None
            
            # Forward pass
            recon, mu, logvar = model(data)
            
            # Compute loss
            loss, recon_loss, kl_loss = compute_loss(model, data, recon, mu, logvar, mask, kl_weight)
            
            # Update running losses
            val_losses['total'] += loss.item()
            val_losses['recon'] += recon_loss.item()
            val_losses['kl'] += kl_loss.item()
    
    # Calculate average losses
    avg_losses = {k: v / num_batches for k, v in val_losses.items()}
    
    return avg_losses

def save_checkpoint(model, optimizer, epoch, losses, config, filepath):
    """Save a complete checkpoint including model, optimizer state, and metadata"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'config': config
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {Colors.GREEN}{filepath}{Colors.RESET}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load a checkpoint with error handling"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu')  # Load to CPU first
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {Colors.GREEN}{filepath}{Colors.RESET}")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {Colors.RED}{e}{Colors.RESET}")
        return None

def plot_training_curves(losses, output_path='training_curves.png', model_name=None, num_params=None):
    """Plot and save training curves"""
    plt.figure(figsize=(12, 8))
    
    # Add title with model name and parameter count if provided
    title = "Training Curves"
    if model_name:
        title = f"{model_name} VAE Training Curves"
    if num_params:
        title += f" ({num_params:,} parameters)"
    plt.suptitle(title, fontsize=16)
    
    plt.subplot(2, 1, 1)
    plt.plot(losses['train_total'], label='Training Loss')
    if 'val_total' in losses and losses['val_total']:
        plt.plot(losses['val_total'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Set y-axis to log scale
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(losses['train_recon'], label='Recon (Train)', color='blue')
    plt.plot(losses['train_kl'], label='KL (Train)', color='green')
    if 'val_recon' in losses and losses['val_recon']:
        plt.plot(losses['val_recon'], label='Recon (Val)', color='cyan')
        plt.plot(losses['val_kl'], label='KL (Val)', color='limegreen')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Set y-axis to log scale
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Training curves saved to {Colors.GREEN}{output_path}{Colors.RESET}")

def train_model(model_type, model_params, train_params, data_dir, device):
    """Train a specific ConvNextVAE model based on parameters"""
    
    model_name = model_type.capitalize()
    logger.info(f"Starting training for {Colors.BOLD}{Colors.GREEN}Model: {model_name}{Colors.RESET}")
    
    # Create model-specific checkpoint directory
    checkpoint_dir = Path(f"checkpoints/{model_type}_vae")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get appropriate dataloader based on model type
    if model_type == 'translation':
        dataset = BobTranslationDataset(data_dir, person_idx=0)
        train_loader = DataLoader(
            dataset,
            batch_size=train_params['batch_size'],
            shuffle=True,
            num_workers=train_params.get('num_workers', 0)
        )
    elif model_type == 'orientation':
        dataset = BobOrientationDataset(data_dir, person_idx=0)
        train_loader = DataLoader(
            dataset,
            batch_size=train_params['batch_size'],
            shuffle=True,
            num_workers=train_params.get('num_workers', 0)
        )
    elif model_type == 'pose':
        dataset = BobPoseDataset(data_dir, person_idx=0)
        train_loader = DataLoader(
            dataset,
            batch_size=train_params['batch_size'],
            shuffle=True,
            num_workers=train_params.get('num_workers', 0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Important: Model's T_in parameter must match the temporal dimension of the dataset
    # Our datasets return sequences of length 100, so model_params['T_in'] should be 100
    # If there's a mismatch, the model will either raise a validation error or try to adjust
    # itself during the forward pass, which can lead to unexpected behavior
    
    # Create specific ConvNextVAE for this motion aspect
    model = ConvNextVAE(
        latent_dim=model_params.get('latent_dim', 128),
        T_in=model_params.get('T_in', 100),
        input_dim=model_params.get('input_dim', 138),
        encoder_output_channels=model_params.get('encoder_output_channels', 512),
        down_t=model_params.get('down_t', 2),
        stride_t=model_params.get('stride_t', 2),
        width=model_params.get('width', 512),
        depth=model_params.get('depth', 3),
    ).to(device)
    
    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {Colors.YELLOW}{num_params:,}{Colors.RESET} parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_params['learning_rate'], 
        weight_decay=train_params.get('weight_decay', 1e-4)
    )
    
    # Initialize tracking variables
    num_epochs = train_params['epochs']
    best_loss = float('inf')
    losses = {
        'train_total': [], 'train_recon': [], 'train_kl': [],
        'val_total': [], 'val_recon': [], 'val_kl': []
    }
    
    # Initialize timer
    timer = Timer().start()
    
    # Training loop
    logger.info(f"{Colors.BOLD}{Colors.GREEN}Starting training for {num_epochs} epochs{Colors.RESET}")
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_losses = train_epoch(
            model, train_loader, optimizer, device,
            kl_weight=train_params.get('kl_loss_weight', 0.01),
            epoch=epoch, total_epochs=num_epochs, timer=timer
        )
        
        # Record losses
        losses['train_total'].append(train_losses['total'])
        losses['train_recon'].append(train_losses['recon'])
        losses['train_kl'].append(train_losses['kl'])
        
        # Log epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {Colors.YELLOW}{epoch+1}/{num_epochs}{Colors.RESET} - "
                  f"{Colors.MAGENTA}Loss: {train_losses['total']:.4f}{Colors.RESET} "
                  f"({Colors.CYAN}Recon: {train_losses['recon']:.4f}{Colors.RESET}, "
                  f"{Colors.GREEN}KL: {train_losses['kl']:.4f}{Colors.RESET}) - "
                  f"{Colors.BLUE}LR: {current_lr:.6f}{Colors.RESET}")
        
        # Save intermediate checkpoints
        if (epoch + 1) % train_params.get('save_interval', 10) == 0:
            checkpoint_path = checkpoint_dir / f"{model_type}_vae_epoch_{epoch+1}.pt"
            save_checkpoint(model, optimizer, epoch, losses, {'model_params': model_params, 'train_params': train_params}, checkpoint_path)
        
        # Save the best model
        if train_losses['total'] < best_loss:
            best_loss = train_losses['total']
            best_model_path = checkpoint_dir / f"{model_type}_vae_best.pt"
            save_checkpoint(model, optimizer, epoch, losses, {'model_params': model_params, 'train_params': train_params}, best_model_path)
            logger.info(f"{Colors.BOLD}{Colors.GREEN}New best model!{Colors.RESET} "
                       f"Loss: {Colors.MAGENTA}{best_loss:.4f}{Colors.RESET}")
            
        # Plot and save training curves periodically
        if (epoch + 1) % train_params.get('plot_interval', 10) == 0:
            plot_training_curves(
                losses, 
                output_path=checkpoint_dir / f"{model_type}_training_curves_epoch_{epoch+1}.png",
                model_name=model_name,
                num_params=num_params
            )
    
    # Save final model
    final_model_path = checkpoint_dir / f"{model_type}_vae_final.pt"
    save_checkpoint(model, optimizer, num_epochs-1, losses, {'model_params': model_params, 'train_params': train_params}, final_model_path)
    
    # Save just the model weights for easier loading
    torch.save(model.state_dict(), checkpoint_dir / f"{model_type}_vae_weights_final.pt")
    
    # Final training curves
    plot_training_curves(
        losses, 
        output_path=checkpoint_dir / f"{model_type}_training_curves_final.png",
        model_name=model_name,
        num_params=num_params
    )
    
    # Report total training time
    total_time = timer.elapsed()
    logger.info(f"{Colors.BOLD}{Colors.GREEN}Training for {model_name} completed successfully!{Colors.RESET} "
               f"Total time: {Colors.CYAN}{Timer.format_time(total_time)}{Colors.RESET}")
    
    return model, best_loss

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train VAE models for motion data')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--train_vae', type=str, default='translation,orientation,pose', 
                      help='Comma-separated list of VAEs to train (translation,orientation,pose)')
    args = parser.parse_args()
    
    # Parse which VAEs to train
    vae_types_to_train = [vae_type.strip().lower() for vae_type in args.train_vae.split(',')]
    valid_vae_types = ['translation', 'orientation', 'pose']
    
    # Validate VAE types
    for vae_type in vae_types_to_train:
        if vae_type not in valid_vae_types:
            logger.error(f"Invalid VAE type: {vae_type}. Valid types are: {', '.join(valid_vae_types)}")
            return
    
    logger.info(f"Training the following VAE types: {', '.join(vae_types_to_train)}")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {Colors.BOLD}{Colors.GREEN}{device}{Colors.RESET}")
    
    # Define hyperparameters for each model type
    
    # 1. Translation VAE parameters
    translation_model_params = {
        'latent_dim': 64,                # Smaller latent space for translation
        'T_in': 100,                     # Change from 120 to 100 to match the dataset
        'input_dim': 3,                  # x, y, z coordinates
        'encoder_output_channels': 128,  # Smaller channel size
        'down_t': 3,                     # More aggressive downsampling
        'stride_t': 2,
        'width': 128,                    # Smaller width
        'depth': 2,                      # Fewer layers
    }
    
    translation_train_params = {
        'epochs': 100,                   # Fewer epochs for translation
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'kl_loss_weight': 0.005,         # Lower KL weight
        'save_interval': 10,
        'plot_interval': 10
    }
    
    # 2. Orientation VAE parameters
    orientation_model_params = {
        'latent_dim': 96,                # Medium latent space for orientation
        'T_in': 100,                     # Change from 120 to 100 to match the dataset
        'input_dim': 3,                  # Changed from 6 to 3 to match the actual data dimensions
        'encoder_output_channels': 128,  # Medium channel size
        'down_t': 2,
        'stride_t': 2,
        'width': 128,                    # Medium width
        'depth': 2,                      # Medium depth
    }
    
    orientation_train_params = {
        'epochs': 200,                   # Medium number of epochs
        'batch_size': 64,
        'learning_rate': 8e-5,
        'weight_decay': 1e-5,
        'kl_loss_weight': 0.008,
        'save_interval': 10,
        'plot_interval': 10
    }
    
    # 3. Body Pose VAE parameters
    pose_model_params = {
        'latent_dim': 128,
        'T_in': 100,
        'input_dim': 63, 
        'encoder_output_channels': 64,
        'down_t': 2,
        'stride_t': 2,
        'width': 256,
        'depth': 3,
    }
    
    pose_train_params = {
        'epochs': 200,                   # More epochs for pose
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'kl_loss_weight': 0.05,
        'save_interval': 10,
        'plot_interval': 20
    }
    
    # Track best losses for each model
    best_losses = {}
    
    # Train only the specified VAEs
    if 'translation' in vae_types_to_train:
        translation_model, translation_loss = train_model(
            model_type='translation',
            model_params=translation_model_params,
            train_params=translation_train_params,
            data_dir=args.data_dir,
            device=device
        )
        best_losses['translation'] = translation_loss
    
    if 'orientation' in vae_types_to_train:
        orientation_model, orientation_loss = train_model(
            model_type='orientation',
            model_params=orientation_model_params,
            train_params=orientation_train_params,
            data_dir=args.data_dir,
            device=device
        )
        best_losses['orientation'] = orientation_loss
    
    if 'pose' in vae_types_to_train:
        pose_model, pose_loss = train_model(
            model_type='pose',
            model_params=pose_model_params,
            train_params=pose_train_params,
            data_dir=args.data_dir,
            device=device
        )
        best_losses['pose'] = pose_loss
    
    # Log best losses for trained models
    logger.info("===== Training Summary =====")
    for model_type, loss in best_losses.items():
        logger.info(f"Best {model_type.capitalize()} Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
