from typing import Dict, Any, Type, Union
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Optimizer
import torch
from model.convnext import ConvNextVAE
from model.transformer_vae import TransformerVAE
from model.hierarchical_transformer import HierarchicalTransformerVAE
from model.videovaeplus import VideoVAEPlus
from utils.loss import (
    combined_motion_loss, 
    geodesic_loss, 
    velocity_loss,
)
from dataset.solo import (
    SoloTranslationDataset,
    SoloOrientationDataset,
    SoloPoseDataset,
    SoloPose6DDataset,
    SoloFullMotionDataset,
)

from utils.loss import combined_motion_loss, geodesic_loss, velocity_loss

#------------------------------------------------------------------------------------------------
# Model factory
#------------------------------------------------------------------------------------------------

def get_model(name: str, config: Dict[str, Any] = None) -> nn.Module:
    """Get model by name with configuration."""
    models: Dict[str, Type[nn.Module]] = {
        'convnext': ConvNextVAE,
        'transformer': TransformerVAE,
        'hierarchical_transformer': HierarchicalTransformerVAE,
        'videovaeplus': VideoVAEPlus
    }
    
    if name not in models:
        raise ValueError(f"Model {name} not found. Available models: {list(models.keys())}")
    
    return models[name](**config) if config else models[name]()

def load_model(name: str, config: Dict[str, Any] = None) -> nn.Module:
    """Load model by name with configuration."""
    model = get_model(name, config['model']['params'])
    ckpt_path = f"outputs/{name}_{config['dataset']['name']}/checkpoints/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model

#------------------------------------------------------------------------------------------------
# Dataset factory
#------------------------------------------------------------------------------------------------

def get_dataset(name: str, **kwargs) -> Dataset:
    """Get dataset by name."""
    datasets: Dict[str, Type[Dataset]] = {
        'solo_translation': SoloTranslationDataset,
        'solo_orientation': SoloOrientationDataset,
        'solo_pose': SoloPoseDataset,
        'solo_pose_6d': SoloPose6DDataset,
        'solo_full': SoloFullMotionDataset,
    }
    
    if name not in datasets:
        raise ValueError(f"Dataset {name} not found. Available datasets: {list(datasets.keys())}")
    
    return datasets[name](**kwargs)


#------------------------------------------------------------------------------------------------
# Loss factory
#------------------------------------------------------------------------------------------------

def get_loss(name: str, **kwargs) -> callable:
    """Get loss function by name."""
    losses = {
        'combined': combined_motion_loss,
        'geodesic': geodesic_loss,
        'velocity': velocity_loss,
        'mse': F.mse_loss,
        'l1': F.l1_loss
    }
    
    if name not in losses:
        raise ValueError(f"Loss {name} not found. Available losses: {list(losses.keys())}")
    
    return losses[name]


#------------------------------------------------------------------------------------------------
# Optimizer factory
#------------------------------------------------------------------------------------------------

def get_optimizer(name: str, parameters, **kwargs) -> Optimizer:
    """Get optimizer by name."""
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop
    }
    
    if name not in optimizers:
        raise ValueError(f"Optimizer {name} not found. Available optimizers: {list(optimizers.keys())}")
    
    return optimizers[name](parameters, **kwargs)

