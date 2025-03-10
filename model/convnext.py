"""
ConvNeXt-based VAE implementation for motion sequence modeling.
This module provides a 1D adaptation of the ConvNeXt architecture for VAE-based 
motion sequence generation and reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

#------------------------------------------------------------------------------------------------
# DropPath Implementation
#------------------------------------------------------------------------------------------------

def drop_path(x: torch.Tensor, 
              drop_prob: float = 0., 
              training: bool = False, 
              scale_by_keep: bool = True) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks.
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping a path
        training: Whether model is in training mode
        scale_by_keep: Whether to scale outputs by keep probability
        
    Returns:
        Output tensor after applying drop path
    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with different dim tensors
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
        
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks.
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'
    

#------------------------------------------------------------------------------------------------
# ConvNeXt Block for 1D (Temporal) Data
#------------------------------------------------------------------------------------------------

class ConvNextBlock1D(nn.Module):
    """
    A ConvNeXt block adapted for 1D (temporal) signals.
    
    Architecture:
    1. Depthwise convolution
    2. Layer normalization
    3. Pointwise expansion
    4. GELU activation
    5. Pointwise reduction
    6. Drop path (optional)
    7. Residual connection
    """
    def __init__(
        self, 
        dim: int, 
        kernel_size: int = 7, 
        expansion: int = 4, 
        drop_path: float = 0.0
    ):
        """
        Args:
            dim: Number of input/output channels
            kernel_size: Kernel size for the depthwise convolution
            expansion: Expansion factor for the pointwise layers
            drop_path: Drop path rate for stochastic depth
        """
        super().__init__()
        # Depthwise convolution
        self.dwconv = nn.Conv1d(
            dim, 
            dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=dim
        )
        
        # Remaining layers
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)  # Expand channels
        self.activation = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)  # Reduce channels back
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvNextBlock.
        
        Args:
            x: Input tensor of shape (B, C, T) in channels-first format
            
        Returns:
            Output tensor of shape (B, C, T)
        """
        residual = x
        
        # Depthwise convolution
        x = self.dwconv(x)
        
        # Channel-wise processing
        x = x.transpose(1, 2)  # (B, T, C) for LayerNorm and Linear
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # Back to (B, C, T)
        
        # Drop path and residual connection
        x = self.drop_path(x)
        x = residual + x
        
        return x


class ConvNext1D(nn.Module):
    """
    Stack of ConvNextBlock1D layers.
    """
    def __init__(self, n_channels: int, depth: int):
        """
        Args:
            n_channels: Number of input/output channels
            depth: Number of ConvNextBlock1D layers
        """
        super().__init__()
        self.model = nn.Sequential(*[
            ConvNextBlock1D(
                dim=n_channels, 
                kernel_size=7, 
                expansion=4, 
                drop_path=0.0
            ) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the stack of ConvNextBlock1D layers.
        
        Args:
            x: Input tensor of shape (B, C, T)
            
        Returns:
            Output tensor of shape (B, C, T)
        """
        return self.model(x)


#------------------------------------------------------------------------------------------------
# Encoder and Decoder
#------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Encoder network for the VAE, consisting of temporal downsampling and ConvNext blocks.
    """
    def __init__(
        self,
        input_emb_width: int,     # Number of features per frame
        enc_emb_width: int,       # Number of channels output by the encoder
        down_t: int,              # Number of downsampling blocks
        stride_t: int,            # Stride of the downsampling blocks
        width: int,               # Width of the encoder
        depth: int,               # Depth of each ConvNext stack
    ):
        super().__init__()
        
        # Build the encoder blocks
        blocks: List[nn.Module] = []
        
        # Initial projection to width channels
        blocks.append(nn.Conv1d(input_emb_width, width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.ReLU())
        
        # Downsampling blocks
        filter_t, pad_t = stride_t * 2, stride_t // 2
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                # Downsample with strided convolution
                nn.Conv1d(width, width, kernel_size=filter_t, stride=stride_t, padding=pad_t),
                # Process with ConvNext blocks
                ConvNext1D(width, depth)
            ))
        
        # Final projection to output embedding width
        blocks.append(nn.Conv1d(width, enc_emb_width, kernel_size=3, stride=1, padding=1))
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute to channels-first format for convolutions
        x = x.permute(0, 2, 1)  # (B, input_emb_width, T)
        return self.model(x)    # (B, enc_emb_width, T_out)


class Decoder(nn.Module):
    """
    Decoder network for the VAE, consisting of temporal upsampling and ConvNext blocks.
    """
    def __init__(
        self,
        T_out: int,               # Number of time frames in output
        enc_emb_width: int,       # Number of channels from the encoder
        output_emb_width: int,    # Number of features per output frame
        down_t: int,              # Number of upsampling blocks (matching encoder)
        stride_t: int,            # Stride factor for upsampling
        width: int,               # Width of the decoder
        depth: int,               # Depth of each ConvNext stack
    ):
        super().__init__()
        
        # Build the decoder blocks
        blocks: List[nn.Module] = []
        
        # Initial projection to width channels
        blocks.append(nn.Conv1d(enc_emb_width, width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.ReLU())
        
        # Upsampling blocks
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                # Process with ConvNext blocks
                ConvNext1D(width, depth),
                # Upsample with nearest neighbor (simpler than transposed conv)
                nn.Upsample(scale_factor=stride_t, mode='nearest')
            ))
        
        # Additional processing and final projection
        blocks.append(nn.Conv1d(width, width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, output_emb_width, kernel_size=3, stride=1, padding=1))
        
        # Ensure exact output size with final upsampling
        blocks.append(nn.Upsample(size=T_out, mode='linear', align_corners=False))
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process in channels-first format
        x = self.model(x)  # (B, output_emb_width, T_out)
        
        # Permute back to sequence-first format
        return x.permute(0, 2, 1)  # (B, T_out, output_emb_width)


#------------------------------------------------------------------------------------------------
# ConvNext VAE
#------------------------------------------------------------------------------------------------

class ConvNextVAE(nn.Module):
    """
    Variational Autoencoder using ConvNeXt blocks for motion sequence modeling.
    """
    def __init__(
        self, 
        latent_dim: int = 128, 
        T_in: int = 100, 
        input_dim: int = 138,
        encoder_output_channels: int = 512, 
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
    ):
        """
        Args:
            latent_dim: Dimension of the latent space
            T_in: Number of time frames in input sequence
            input_dim: Number of features per frame
            encoder_output_channels: Number of channels in encoder output
            down_t: Number of downsampling/upsampling blocks
            stride_t: Stride factor for downsampling/upsampling
            width: Width of the ConvNext blocks
            depth: Depth of each ConvNext stack
            use_body_encoding: Whether to use body encoding
        """
        super().__init__()
        self.masked_frame = nn.Parameter(torch.randn(1, input_dim))
        self.body_encoding = nn.Parameter(torch.zeros(1, input_dim))

        self.latent_dim = latent_dim
        self.T_in = T_in
        
        # Calculate temporal dimension after downsampling
        T_current = T_in
        for _ in range(down_t):
            # Formula for conv output size: ((input_size + 2*padding - kernel_size) / stride) + 1
            filter_t, stride, pad_t = stride_t * 2, stride_t, stride_t // 2
            T_current = ((T_current + 2*pad_t - filter_t) // stride) + 1
        
        self.T_out = T_current
        self.flattened_size = encoder_output_channels * self.T_out
        
        # Track whether we've already adjusted the linear layers
        self.linear_layers_adjusted = False

        # Build encoder
        self.encoder = Encoder(
            input_emb_width=input_dim,
            enc_emb_width=encoder_output_channels,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
        )

        # Latent space projections
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)

        # Build decoder
        self.decoder = Decoder(
            T_out=T_in,
            enc_emb_width=encoder_output_channels,
            output_emb_width=input_dim,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence to latent representation.
        
        Args:
            x: Input tensor of shape (B, T, input_dim)
            
        Returns:
            Tuple of (mu, logvar) each of shape (B, latent_dim)
        """
            
        # Encode sequence
        h = self.encoder(x)  # (B, encoder_output_channels, T_out)
        
        # Check if actual T_out matches expected T_out and adjust if needed
        actual_t_out = h.size(2)
        if actual_t_out != self.T_out and not self.linear_layers_adjusted:
            print(f"Warning: Expected T_out={self.T_out}, but got actual T_out={actual_t_out}. Adjusting flattened_size.")
            self.T_out = actual_t_out
            actual_flattened_size = h.size(1) * actual_t_out
            
            # Only recreate linear layers if the size actually changed
            if actual_flattened_size != self.flattened_size:
                self.flattened_size = actual_flattened_size
                # Recreate the linear layers with the correct dimensions
                device = self.fc_mu.weight.device
                self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim).to(device)
                self.fc_logvar = nn.Linear(self.flattened_size, self.latent_dim).to(device)
                self.fc_decode = nn.Linear(self.latent_dim, self.flattened_size).to(device)
                print(f"Adjusted linear layers to use flattened_size={self.flattened_size}")
            
            # Set the flag to avoid readjusting on subsequent forward passes
            self.linear_layers_adjusted = True
        
        # Flatten and project to latent parameters
        h = h.view(x.size(0), -1)  # (B, flattened_size)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Perform the reparameterization trick to sample from the latent distribution.
        
        Args:
            mu: Mean tensor of shape (B, latent_dim)
            logvar: Log variance tensor of shape (B, latent_dim)
            
        Returns:
            Sampled latent vector of shape (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output sequence.
        
        Args:
            z: Latent vector of shape (B, latent_dim)
            
        Returns:
            Reconstructed sequence of shape (B, T, input_dim)
        """
        # Project from latent space to flattened representation
        h = self.fc_decode(z)
        
        # Reshape to match encoder output shape
        h = h.view(z.size(0), -1, self.T_out)  # (B, encoder_output_channels, T_out)
        
        # Decode to sequence
        return self.decoder(h)  # (B, T, input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor of shape (B, T, input_dim)
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        if mask is not None:
            x[mask] = self.masked_frame
        # Validate input dimensions
        if x.size(1) != self.T_in:
            raise ValueError(f"Input temporal dimension {x.size(1)} does not match expected T_in={self.T_in}. "
                             f"Please ensure the model T_in parameter matches the dataset's sequence length.")
        
        # Encode to latent space
        mu, logvar = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode to reconstruction
        recon = self.decode(z)
        
        return recon, mu, logvar

    def compute_loss(self, x: torch.Tensor, recon_x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor,
                     kl_weight_override: Optional[float] = None,
                     mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss components: reconstruction loss and KL divergence.
        
        Args:
            x: Original input tensor
            recon_x: Reconstructed tensor
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            kl_weight_override: Optional override for KL weight
            mask: Optional boolean mask tensor where False indicates masked values to ignore
            
        Returns:
            Tuple of (total_loss, recon_loss, kl_loss)
        """
        # Default KL weight if not overridden
        kl_weight = 0.01 if kl_weight_override is None else kl_weight_override
        
        # Reconstruction loss (MSE)
        if mask is not None:
            # Apply mask to consider only valid parts in loss
            valid_x = x[mask]
            valid_recon_x = recon_x[mask]
            recon_loss = F.mse_loss(valid_recon_x, valid_x, reduction='mean')
        else:
            recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss

    
def main():
    """
    Test a forward pass through the ConvNextVAE network.
    """
    import torch
    
    # Define model parameters
    model_params = {
        'latent_dim': 64,
        'T_in': 100,
        'input_dim': 3,
        'encoder_output_channels': 128,
        'down_t': 2,
        'stride_t': 2,
        'width': 128,
        'depth': 2,
    }
    
    # Create model
    model = ConvNextVAE(**model_params)
    print(f"Created ConvNextVAE with parameters: {model_params}")
    
    # Create random input tensor
    batch_size = 4
    x = torch.randn(batch_size, model_params['T_in'], model_params['input_dim'])
    mask = torch.bernoulli(torch.ones(batch_size, model_params['T_in']) * 0.5).bool()
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    # Forward pass
    with torch.no_grad():
        recon, mu, logvar = model(x, mask)
    
    # Print shapes
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Compute loss
    total_loss, recon_loss, kl_loss = model.compute_loss(x, recon, mu, logvar)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    print("Forward pass test completed successfully!")

if __name__ == "__main__":
    main()

