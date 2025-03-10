import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder, XTransformer
import math
from torch.amp import autocast
from typing import Optional

class HierarchicalTransformerVAE(nn.Module):
    def __init__(
        self,
        input_dim,          
        timesteps,          
        latent_dim,         
        depth=4,
        heads=8,
        dropout=0.1,
        kl_weight=0.005,
        use_cache=True      # Enable attention caching
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.use_cache = use_cache
        
        # Shared encoder configuration for efficiency
        transformer_config = dict(
            heads=heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            ff_mult=4,
            ff_glu=True,            
            rotary_pos_emb=True,    
        )
        
        # Encoder components
        self.register_buffer('cls_token', torch.randn(1, 1, latent_dim))
        
        # Spatial encoding branch
        self.enc_spatial_proj = nn.Linear(timesteps, latent_dim)
        self.enc_spatial_norm = nn.LayerNorm(latent_dim)
        self.spatial_encoder = Encoder(
            dim=latent_dim,
            depth=depth,
            rel_pos_bias=False,
            **transformer_config
        )
        
        # Temporal encoding branch
        self.enc_temporal_proj = nn.Linear(input_dim, latent_dim)
        self.enc_temporal_norm = nn.LayerNorm(latent_dim)
        self.temporal_encoder = Encoder(
            dim=latent_dim,
            depth=depth,
            rel_pos_bias=False,
            alibi_pos_bias=True,
            **transformer_config
        )
        
        # VAE components
        self.to_latent = nn.Linear(latent_dim, latent_dim * 2)
        
        # Decoder components
        # Initial projection from latent space
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * latent_dim),
            nn.LayerNorm(latent_dim * latent_dim),
            nn.GELU()
        )
        
        # Temporal decoding branch
        self.dec_temporal_norm = nn.LayerNorm(latent_dim)
        self.temporal_decoder = Encoder(
            dim=latent_dim,
            depth=depth,
            rel_pos_bias=False,
            alibi_pos_bias=True,
            **transformer_config
        )
        self.dec_temporal_proj = nn.Linear(latent_dim, input_dim)
        
        # Spatial decoding branch
        self.dec_spatial_proj = nn.Linear(latent_dim, latent_dim)
        self.dec_spatial_norm = nn.LayerNorm(latent_dim)
        self.spatial_decoder = Encoder(
            dim=latent_dim,
            depth=depth,
            rel_pos_bias=False,
            alibi_pos_bias=True,
            **transformer_config
        )
        self.dec_spatial_proj = nn.Linear(latent_dim, timesteps)
        
        # Final output processing
        self.to_output = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.Tanh()
        )
        
        # KL weight parameters
        self.register_buffer('kld_weight', torch.tensor(kl_weight))
        self.register_buffer('min_kl_weight', torch.tensor(kl_weight / 10))
        self.register_buffer('max_kl_weight', torch.tensor(kl_weight))
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.2)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @autocast(device_type='cuda')
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        # Spatial encoding
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.enc_spatial_proj(x)  # Project to latent dim
        x = self.enc_spatial_norm(x)
        x = self.spatial_encoder(x)
        
        # Temporal encoding
        x = x.transpose(1, 2)  # [B, T, D]
        x = self.enc_temporal_proj(x)
        x = self.enc_temporal_norm(x)
        
        # Add CLS token for global features
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.temporal_encoder(x)
        
        # Get latent parameters from CLS token
        x = x[:, 0]  # Take CLS token features
        latent_params = self.to_latent(x)
        mu, logvar = latent_params.chunk(2, dim=-1)
        
        return mu, logvar

    @autocast(device_type='cuda')
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        
        # Project latent to initial sequence
        x = self.latent_proj(z)
        x = x.reshape(batch_size, self.latent_dim, self.latent_dim)
        
        # Temporal decoding
        x = self.dec_temporal_norm(x)
        x = self.temporal_decoder(x)
        x = self.dec_temporal_proj(x)
        x = x.transpose(1, 2)
        
        # Spatial decoding
        x = self.dec_spatial_norm(x)
        x = self.spatial_decoder(x)
        x = self.dec_spatial_proj(x)
        x = x.transpose(1, 2)
        
        
        # Add shape assertion
        assert x.shape[1] == self.timesteps, f"Expected sequence length {self.timesteps}, got {x.shape[1]}"
        
        return x

    @autocast(device_type='cuda')
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input normalization with better numerical stability
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_normalized = (x - x_mean) / x_std
        
        # Encode-decode pipeline
        mu, logvar = self.encode(x_normalized)
        z = self.reparameterize(mu, logvar)
        recon_normalized = self.decode(z)
        
        # Efficient residual connection
        if self.training:
            scale = torch.sigmoid(self.residual_scale)
            recon_normalized = torch.lerp(recon_normalized, x_normalized, scale)
        
        # Denormalize
        recon = recon_normalized * x_std + x_mean
        
        return recon, mu, logvar

    def compute_loss(
        self, 
        x: torch.Tensor, 
        recon: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor, 
        kl_weight_override: Optional[float] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Efficient loss computation
        recon_loss = 0.5 * (
            F.l1_loss(recon, x, reduction='mean') +
            F.mse_loss(recon, x, reduction='mean')
        )
        
        # Numerically stable KL divergence
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        
        kl_weight = self.kld_weight if kl_weight_override is None else kl_weight_override
        loss = recon_loss + kl_weight * kld_loss
        
        return loss, recon_loss, kld_loss

    @torch.no_grad()
    def update_kl_weight(self, epoch: int, total_epochs: int) -> float:
        # Efficient KL weight update
        progress = min(1.0, epoch / (0.75 * total_epochs))
        self.kld_weight = torch.lerp(self.min_kl_weight, self.max_kl_weight, progress)
        return self.kld_weight.item()


# --- Example Usage ---
if __name__ == '__main__':
    B, T, D = 8, 100, 63
    latent_dim = 128  # Increased latent dimension for better reconstruction
    model = HierarchicalTransformerVAE(input_dim=D, latent_dim=latent_dim, timesteps=T)

    # Create dummy input data
    x = torch.randn(B, T, D)
    recon, mu, logvar = model(x)
    
    # Compute loss
    loss, recon_loss, kld_loss = model.compute_loss(x, recon, mu, logvar)
    
    print(f"Reconstructed shape: {recon.shape}")
    print(f"Latent parameters shape: {mu.shape}, {logvar.shape}")
    print(f"Total loss: {loss.item():.4f}, Recon loss: {recon_loss.item():.4f}, KLD: {kld_loss.item():.4f}")
