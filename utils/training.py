import logging
import time
from datetime import timedelta, datetime
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from typing import Dict

# Constants should be in UPPER_CASE
class COLORS:
    """ANSI color codes for terminal output."""
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


class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored log output."""
    
    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt)
        timestamp = f"{COLORS.CYAN}{timestamp}{COLORS.RESET}"
        
        level_colors = {
            'INFO': COLORS.GREEN,
            'WARNING': COLORS.YELLOW,
            'ERROR': COLORS.RED,
            'CRITICAL': f"{COLORS.BG_RED}{COLORS.WHITE}"
        }
        
        levelname = record.levelname
        color = level_colors.get(levelname, '')
        levelname = f"{color}{levelname}{COLORS.RESET}"
        
        name = f"{COLORS.BLUE}{record.name}{COLORS.RESET}"
        msg = self._colorize_message(record.getMessage())
        
        return f"{timestamp} - {name} - {levelname} - {msg}"
    
    def _colorize_message(self, msg):
        """Add colors to specific keywords in the message."""
        color_mappings = {
            "Epoch": COLORS.YELLOW,
            "Loss:": f"{COLORS.BOLD}{COLORS.MAGENTA}",
            "(Recon:": f"({COLORS.CYAN}",
            "KL:": COLORS.GREEN,
            "LR:": f"{COLORS.BOLD}{COLORS.BLUE}",
            "Model:": f"{COLORS.BOLD}{COLORS.WHITE}"
        }
        
        for keyword, color in color_mappings.items():
            if keyword in msg:
                msg = msg.replace(keyword, f"{color}{keyword}{COLORS.RESET}")
        return msg


class Timer:
    """Utility class for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.lap_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.lap_time = self.start_time
        return self
    
    def lap(self):
        """Return time elapsed since last lap."""
        current_time = time.time()
        elapsed = current_time - self.lap_time
        self.lap_time = current_time
        return elapsed
    
    def elapsed(self):
        """Return total elapsed time."""
        return time.time() - self.start_time
    
    @staticmethod
    def format_time(seconds):
        """Format seconds into human-readable time string."""
        return str(timedelta(seconds=int(seconds)))


def setup_logger(name='vae_training'):
    """Set up and configure logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    # Remove existing handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_dir / f"vae_training_{timestamp}.log"
    )
    file_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(model, optimizer, epoch, losses, config, filepath, logger):
    """Save a complete checkpoint including model state and metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'config': config
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {COLORS.GREEN}{filepath}{COLORS.RESET}")


def load_checkpoint(filepath, model, optimizer, logger):
    """Load a checkpoint with error handling."""
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {COLORS.GREEN}{filepath}{COLORS.RESET}")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {COLORS.RED}{e}{COLORS.RESET}")
        return None


def plot_training_curves(
    losses: Dict, 
    output_path: str = 'training_curves.png',
    model_name: str = None,
    dataset_name: str = None,
    num_params: int = None,
    logger: logging.Logger = None
) -> None:
    """Plot and save training curves with loss components."""
    plt.figure(figsize=(12, 8))
    
    # Title
    title = "Training Curves"
    if model_name:
        title = f"{model_name} VAE for {dataset_name}"
    if num_params:
        title += f" ({num_params:,} parameters)"
    plt.suptitle(title, fontsize=16)
    
    # Total loss plot
    plt.subplot(2, 1, 1)
    plt.plot(losses['train_total'], label='Training Loss')
    if 'val_total' in losses and losses['val_total']:
        plt.plot(losses['val_total'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    # Component losses plot
    plt.subplot(2, 1, 2)
    plt.plot(losses['train_recon'], label='Recon (Train)', color='blue')
    plt.plot(losses['train_kl'], label='KL (Train)', color='green')
    if 'val_recon' in losses and losses['val_recon']:
        plt.plot(losses['val_recon'], label='Recon (Val)', color='cyan')
        plt.plot(losses['val_kl'], label='KL (Val)', color='limegreen')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    if logger:
        logger.info(f"Training curves saved to {COLORS.GREEN}{output_path}{COLORS.RESET}")