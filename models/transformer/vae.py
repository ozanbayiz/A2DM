import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder, Decoder

class TransformerVAE(nn.Module):
    def __init__(
        self,
        input_dim,          # feature dimension of each timestep
        latent_dim,         # size of latent vector
        timesteps,          # number of time steps
        encoder_depth=2,    
        decoder_depth=2,
        heads=4,
        dropout=0.1,
        kl_weight=0.01,     # Fixed KL weight (no annealing)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.kl_weight = kl_weight
        
        # --- ENCODER ARCHITECTURE ---
        # Simple transformer encoder
        self.encoder = Encoder(
            dim = input_dim,
            depth = encoder_depth,
            heads = heads,
            attn_dropout = dropout,
            ff_dropout = dropout,
            ff_mult = 4,
            rel_pos_bias = True,
        )
        
        # Simple global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Direct projection to latent parameters
        self.to_mu = nn.Linear(input_dim, latent_dim)
        self.to_logvar = nn.Linear(input_dim, latent_dim)
        
        # --- DECODER ARCHITECTURE ---
        # Simple latent-to-sequence projection
        self.latent_to_sequence = nn.Linear(latent_dim, input_dim * timesteps)
        
        # Simple transformer decoder
        self.decoder = Decoder(
            dim = input_dim,
            depth = decoder_depth,
            heads = heads,
            attn_dropout = dropout,
            ff_dropout = dropout,
            ff_mult = 4,
            cross_attend = False,  # No cross-attention for simplicity
            rel_pos_bias = True,
        )

    def reparameterize(self, mu, logvar):
        """Sample from latent distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encode motion sequence to latent distribution parameters"""
        batch_size = x.shape[0]
        
        # Process sequence through transformer encoder
        encoded = self.encoder(x)  # [B, T, D]
        
        # Simple global pooling
        features = encoded.transpose(1, 2)  # [B, D, T]
        global_features = self.global_pool(features).reshape(batch_size, self.input_dim)
        
        # Project to latent parameters
        mu = self.to_mu(global_features)
        logvar = self.to_logvar(global_features)
        
        return mu, logvar

    def decode(self, z):
        """Decode latent vector to motion sequence"""
        batch_size = z.shape[0]
        
        # Create initial sequence from latent
        seq_flat = self.latent_to_sequence(z)
        seq = seq_flat.reshape(batch_size, self.timesteps, self.input_dim)
        
        # Decode with transformer
        decoded = self.decoder(seq)
        
        return decoded

    def forward(self, x):
        """Full VAE forward pass"""
        # Verify input dimensions
        B, T, D = x.shape
        assert T == self.timesteps, f"Input timesteps ({T}) must match model timesteps ({self.timesteps})"
        assert D == self.input_dim, f"Input dimension ({D}) must match model input_dim ({self.input_dim})"
        
        # Encoding
        mu, logvar = self.encode(x)
        
        # Sampling
        z = self.reparameterize(mu, logvar)
        
        # Decoding
        recon = self.decode(z)
        
        return recon, mu, logvar
    
    def compute_loss(self, x, recon, mu, logvar, kl_weight_override=None):
        """Compute VAE loss"""
        # Simple MSE reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL Divergence loss
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Use provided KL weight or instance weight
        kl_weight = kl_weight_override if kl_weight_override is not None else self.kl_weight
        
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
