import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder, Decoder

class SpatialTransformerVAE(nn.Module):
    def __init__(
        self, 
        input_dim, 
        time_dim,
        latent_dim, 
        depth=4, 
        heads=8, 
        dropout_prob=0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.latent_dim = latent_dim
        self.depth = depth
        self.heads = heads
        self.dropout_prob = dropout_prob
        
        # Channel embeddings
        self.channel_embeddings = nn.Parameter(torch.randn(1, input_dim, 1))

        #Time embeddings
        self.time_embeddings = nn.Parameter(torch.randn(1, 1, time_dim))
        
        # Projection from time to latent dimension
        self.time_to_latent = nn.Linear(time_dim, latent_dim)
        
        # CLS tokens for mu and sigma
        self.cls_mu = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.cls_sigma = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        # Encoder transformer
        self.encoder = Encoder(
            dim=latent_dim,
            depth=depth,
            heads=heads,
            layer_dropout=dropout_prob,
            add_value_residual=True,
        )
        
        # Learnable query tokens for decoding
        self.decoder_queries = nn.Parameter(torch.randn(1, input_dim, latent_dim))
        
        # Decoder transformer (non-autoregressive)
        self.decoder = Encoder(
            dim=latent_dim,
            depth=depth,
            heads=heads,
            cross_attend=True,  # Enable cross-attention
            layer_dropout=dropout_prob,
            cross_attn_tokens_dropout=dropout_prob,
            add_value_residual=True,
        )
        
        # Projection from latent back to time dimension
        self.latent_to_time = nn.Linear(latent_dim, time_dim)
        
    def forward(self, x):
        mu, logvar, z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def encode(self, x):
        batch_size = x.size(0)
        
        # Preprocessing: permute from [B, T, D] to [B, D, T]
        x = x.permute(0, 2, 1)
        
        # Add time embeddings
        x = x + self.time_embeddings
        
        # Add channel embeddings
        x = x + self.channel_embeddings
        
        # Project from time to latent dimension [B, D, T] -> [B, D, L]
        x_proj = self.time_to_latent(x)
        # x_reshaped = x.reshape(batch_size * self.input_dim, self.time_dim)
        # x_proj = self.time_to_latent(x_reshaped)
        # x_proj = x_proj.reshape(batch_size, self.input_dim, self.latent_dim)
        
        # Append CLS tokens for mu and sigma [B, D, L] -> [B, D+2, L]
        cls_mu_batch = self.cls_mu.expand(batch_size, -1, -1)
        cls_sigma_batch = self.cls_sigma.expand(batch_size, -1, -1)
        x_with_cls = torch.cat([cls_mu_batch, cls_sigma_batch, x_proj], dim=1)
        
        # Pass through encoder transformer
        encoded = self.encoder(x_with_cls)
        
        # Extract mu and sigma from CLS tokens
        mu = encoded[:, 0]  # First token is mu
        logvar = encoded[:, 1]  # Second token is sigma/logvar
        
        # Sample latent using reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return mu, logvar, z
    
    def decode(self, z):
        batch_size = z.size(0)
        
        # Expand decoder queries to batch size
        queries = self.decoder_queries.expand(batch_size, -1, -1)
        
        # Reshape z to be used as context for cross-attention
        # Shape: [B, 1, latent_dim]
        z_context = z.unsqueeze(1)
        # z_context = z_context + self.channel_embeddings
        
        # Non-autoregressive decoding with cross-attention to z
        decoded = self.decoder(queries, context=z_context)
        
        # Project back from latent to time dimension
        decoded_reshaped = decoded.reshape(batch_size * self.input_dim, self.latent_dim)
        output = self.latent_to_time(decoded_reshaped)
        output = output.reshape(batch_size, self.input_dim, self.time_dim)
        
        # Permute back to original shape [B, T, D]
        output = output.permute(0, 2, 1)
        
        return output
    
    def sample(self, num_samples=1):
        # Sample from prior N(0, I)
        z = torch.randn(num_samples, self.latent_dim, device=self.cls_mu.device)
        return self.decode(z)
    
    
    def loss_function(self, x, x_recon, mu, logvar, kl_weight=1.0):
        """
        Compute VAE loss: reconstruction loss + KL divergence
        
        Args:
            x: original input
            x_recon: reconstructed input
            mu: mean of latent distribution
            logvar: log variance of latent distribution
            kl_weight: weight for KL divergence term
        """
        # Reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    input_dim = 126    # Number of features
    time_dim = 100    # Sequence length
    latent_dim = 256  # Latent dimension
    
    # Create model instance
    model = SpatialTransformerVAE(
        input_dim=input_dim,
        time_dim=time_dim,
        latent_dim=latent_dim
    ).to(device)
    
    # Create random input tensor
    batch_size = 8
    x = torch.randn(batch_size, time_dim, input_dim).to(device)
    
    # Forward pass
    with torch.no_grad():
        x_recon, mu, logvar = model(x)
    
    # Calculate loss
    total_loss, recon_loss, kl_loss = model.loss_function(x, x_recon, mu, logvar)
    
    # Print shapes and loss
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Sample from prior
    samples = model.sample(num_samples=4)
    print(f"Generated samples shape: {samples.shape}")
    
    return model

if __name__ == "__main__":
    main()