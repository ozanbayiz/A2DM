import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder, Decoder
import math

class TransformerVAE(nn.Module):
    def __init__(
        self,
        input_dim,          # feature dimension of each timestep (e.g., 69)
        latent_dim,         # size of latent vector
        timesteps,          # number of time steps (e.g., 100)
        encoder_depth=4,
        decoder_depth=4,
        heads=8,
        dropout=0.1,
        kl_weight=0.005,    # Reduced KL weight to prioritize reconstruction initially
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        
        # Create positional embedding for better sequence modeling
        self.register_buffer('pos_embedding', self._get_sinusoidal_embedding(timesteps, input_dim))
        
        # Encoder with transformer architecture
        self.encoder = Encoder(
            dim = input_dim,
            depth = encoder_depth,
            heads = heads,
            attn_dropout = dropout,
            ff_dropout = dropout,
            ff_mult = 4,
            rel_pos_bias = True  # Use relative positional bias
        )
        
        # Replace complex temporal pooling with projection from mean-pooled features
        self.latent_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # VAE latent projections with proper initialization
        self.to_mu = nn.Linear(input_dim * 2, latent_dim)
        self.to_logvar = nn.Linear(input_dim * 2, latent_dim)
        
        # Decoder for reconstruction
        self.decoder = Encoder(
            dim = input_dim,
            depth = decoder_depth,
            heads = heads,
            attn_dropout = dropout,
            ff_dropout = dropout,
            ff_mult = 4,
            cross_attend = False,
            rel_pos_bias = True
        )
        
        # Improved latent to sequence projection
        self.latent_to_sequence = nn.Sequential(
            nn.Linear(latent_dim, input_dim * 4),
            nn.LayerNorm(input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim * timesteps)
        )
        
        # Final output processing
        self.output_norm = nn.LayerNorm(input_dim)
        self.output_activation = nn.Tanh()  # Bounded output for stability
        
        # Residual connection with trainable scale
        self.use_residual = True
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.2)
        
        # KL divergence weight with annealing support
        self.kld_weight = kl_weight
        self.min_kl_weight = kl_weight / 10
        self.max_kl_weight = kl_weight

    def reparameterize(self, mu, logvar):
        """Sample from latent distribution with reparameterization trick"""
        # Add clipping for numerical stability
        # logvar = torch.clamp(logvar, -10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encode motion sequence to latent distribution parameters"""
        
        # Process sequence through transformer encoder
        encoded = self.encoder(x)  # [batch_size, timesteps, input_dim]
        
        # Use mean pooling instead of flattening and linear projection
        pooled = encoded.mean(dim=1)  # [batch_size, input_dim]
        
        # Project to higher dimension for better latent representation
        projected = self.latent_projection(pooled)  # [batch_size, input_dim*2]
        
        # Get latent distribution parameters
        mu = self.to_mu(projected)
        logvar = self.to_logvar(projected)
        
        # Add L2 regularization to latent space for stability
        mu = F.normalize(mu, dim=-1, p=2) * math.sqrt(self.latent_dim)
        
        return mu, logvar

    def decode(self, z, x=None):
        """Decode latent vector to motion sequence"""
        batch_size = z.shape[0]
        
        # Project latent to sequence
        seq_flat = self.latent_to_sequence(z)
        seq = seq_flat.reshape(batch_size, self.timesteps, self.input_dim)
        
        # Add positional information
        seq = seq + self.pos_embedding
        
        # Decode with transformer
        decoded = self.decoder(seq)
        
        # Apply output normalization and activation
        output = self.output_norm(decoded)
        output = self.output_activation(output)
        
        # Apply residual connection with learned weight if input is provided
        if self.use_residual and x is not None:
            scale = torch.sigmoid(self.residual_scale)
            output = (1 - scale) * output + scale * x
            
        return output

    def forward(self, x):
        """Full VAE forward pass"""
        B, T, D = x.shape
        assert T == self.timesteps, f"Input timesteps ({T}) must match model timesteps ({self.timesteps})"
        assert D == self.input_dim, f"Input dimension ({D}) must match model input_dim ({self.input_dim})"
        
        # Normalize input data for better stability
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-6
        x_normalized = (x - x_mean) / x_std
        
        # Encoding
        mu, logvar = self.encode(x_normalized)
        
        # Sampling
        z = self.reparameterize(mu, logvar)
        
        # Decoding
        recon_normalized = self.decode(z, x_normalized)
        
        # Denormalize output
        recon = recon_normalized * x_std + x_mean
        
        return recon, mu, logvar
    
    def compute_loss(self, x, recon, mu, logvar, kl_weight_override=None):
        """Compute VAE loss with better balancing between reconstruction and KL divergence"""
        # Use combination of L1 and L2 losses for better reconstruction quality
        recon_loss_l1 = F.l1_loss(recon, x, reduction='mean')
        recon_loss_l2 = F.mse_loss(recon, x, reduction='mean')
        recon_loss = 0.5 * recon_loss_l1 + 0.5 * recon_loss_l2
        
        # KL Divergence loss with improved numerical stability
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Use provided KL weight or instance weight
        kl_weight = kl_weight_override if kl_weight_override is not None else self.kld_weight
        
        # Total loss
        loss = recon_loss + kl_weight * kld_loss
        
        return loss, recon_loss, kld_loss


# --- Example Usage ---
if __name__ == '__main__':
    B, T, D = 8, 100, 70  # 70 features (35 per person)
    latent_dim = 64  # Increased latent dimension for better reconstruction
    model = TransformerVAE(input_dim=D, latent_dim=latent_dim, timesteps=T)

    # Create dummy input data
    x = torch.randn(B, T, D)
    recon, mu, logvar = model(x)
    
    # Compute loss
    loss, recon_loss, kld_loss = model.compute_loss(x, recon, mu, logvar)
    
    print(f"Reconstructed shape: {recon.shape}")
    print(f"Latent parameters shape: {mu.shape}, {logvar.shape}")
    print(f"Total loss: {loss.item():.4f}, Recon loss: {recon_loss.item():.4f}, KLD: {kld_loss.item():.4f}")
