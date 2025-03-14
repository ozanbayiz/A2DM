import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder, Decoder, TransformerWrapper, ContinuousTransformerWrapper

class ACTORVAE_XT(nn.Module):
    def __init__(
        self,
        pose_dim,           # Dimension of pose parameters (SMPL)
        latent_dim=256,     # Dimension of latent space
        hidden_dim=512,     # Transformer hidden dimension
        num_layers=4,       # Number of transformer layers
        num_heads=8,        # Number of attention heads
        dropout=0.1,        # Dropout rate
        max_seq_len=120,    # Maximum sequence length to support
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Special tokens for mean and log-variance
        self.mu_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.logvar_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.input_projection = nn.Linear(pose_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, pose_dim)
        
        # Encoder transformer
        self.encoder = ContinuousTransformerWrapper(
            dim_in=pose_dim,
            dim_out=hidden_dim,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(
                dim=hidden_dim,
                depth=num_layers,
                heads=num_heads,
                attn_dropout=dropout,
                ff_dropout=dropout,
                ff_mult=4,
                ff_glu=True,                # Use GLU variant in feedforward for better performance
                pre_norm=True,              # Use pre-norm for better gradient flow
                attn_head_scale=True,       # From Normformer for better convergence
                use_rmsnorm=True,           # Use RMSNorm for better stability
                rel_pos_bias=True           # Use relative position bias
            )
        )
        
        # Projection from hidden dim to latent space
        self.mu_projection = nn.Linear(hidden_dim, latent_dim)
        self.logvar_projection = nn.Linear(hidden_dim, latent_dim)
        
        # Latent to hidden projection for decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        
        # Query embeddings for decoder (learnable per time step)
        self.query_embeddings = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        
        # Decoder transformer
        self.decoder = ContinuousTransformerWrapper(
            dim_in=hidden_dim,              # Input dimension is hidden_dim for query embeddings
            dim_out=hidden_dim,               # Output dimension is pose_dim directly
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=hidden_dim,
                depth=num_layers,
                heads=num_heads,
                attn_dropout=dropout,
                ff_dropout=dropout,
                ff_mult=4,
                ff_glu=True,                # Use GLU variant in feedforward
                cross_attend=True,          # Enable cross-attention
                pre_norm=True,              # Use pre-norm for better gradient flow
                attn_head_scale=True,       # From Normformer
                use_rmsnorm=True,           # Use RMSNorm
                rel_pos_bias=True           # Use relative position bias
            )
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        # Initialize special tokens
        nn.init.xavier_uniform_(self.mu_token)
        nn.init.xavier_uniform_(self.logvar_token)
        nn.init.xavier_uniform_(self.query_embeddings)
        
    def encode(self, x):
        """
        Encode input motion sequence to latent distribution parameters.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, pose_dim]
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            z: Sampled latent vector
        """
        batch_size, seq_len, _ = x.shape

        # Process input through continuous transformer
        x = self.encoder(
            x,
            prepend_embeds=torch.cat(
                [
                    self.mu_token.expand(batch_size, -1, -1), 
                    self.logvar_token.expand(batch_size, -1, -1)
                ],
                dim=1
            ),
            prepend_mask=torch.ones(batch_size, 2, device=x.device).bool()
        )  # [B, T, H]
        
        mu_hidden = x[:, 0]  # Take first token output [B, H]
        logvar_hidden = x[:, 1]  # Take first token output [B, H]
        
        # Project to latent space
        mu = self.mu_projection(mu_hidden)  # [B, latent_dim]
        logvar = self.logvar_projection(logvar_hidden)  # [B, latent_dim]
        
        # Sample latent vector using reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return mu, logvar, z
    
    def decode(self, z, seq_len=None):
        """
        Decode latent vector to motion sequence.
        
        Args:
            z: Latent vector of shape [batch_size, latent_dim]
            seq_len: Optional sequence length (defaults to max_seq_len)
            
        Returns:
            Reconstructed motion sequence of shape [batch_size, seq_len, pose_dim]
        """
        batch_size = z.shape[0]
        seq_len = seq_len or self.max_seq_len
        
        # Project latent to hidden dimension and prepare as context
        z_hidden = self.latent_to_hidden(z)  # [B, H]
        context = z_hidden.unsqueeze(1)  # [B, 1, H]
        
        # Get query embeddings for desired sequence length
        query_emb = self.query_embeddings[:, :seq_len, :].expand(batch_size, -1, -1)  # [B, seq_len, H]
        
        # Create masks
        query_mask = torch.ones(batch_size, seq_len, device=z.device).bool()
        context_mask = torch.ones(batch_size, 1, device=z.device).bool()
        
        # Use decoder directly with continuous values
        output = self.decoder(
            query_emb,
            mask=query_mask,
            context=context,
            context_mask=context_mask
        )  # [B, seq_len, pose_dim]

        output = self.output_projection(output)
        
        return output
    
    def forward(self, x):
        """
        Forward pass of the VAE.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, pose_dim]
            
        Returns:
            reconstructed: Reconstructed motion sequence
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        # Encode input to latent space
        mu, logvar, z = self.encode(x)
        
        # Decode latent to reconstruction
        seq_len = x.shape[1]
        reconstructed = self.decode(z, seq_len)
        
        return reconstructed, mu, logvar
    
    def sample(self, num_samples=1, seq_len=None, temperature=1.0):
        """
        Sample from the prior distribution to generate new motions.
        
        Args:
            num_samples: Number of samples to generate
            seq_len: Optional sequence length (defaults to max_seq_len)
            temperature: Temperature for sampling (higher = more diverse)
            
        Returns:
            Generated motion sequences of shape [num_samples, seq_len, pose_dim]
        """
        # Sample from prior N(0, I)
        z = torch.randn(num_samples, self.latent_dim, device=self.mu_token.device) * temperature
        
        # Decode to motion sequence
        return self.decode(z, seq_len)
    
    def interpolate(self, z1, z2, num_steps=10, seq_len=None):
        """
        Interpolate between two latent vectors and decode to motion sequences.
        
        Args:
            z1: First latent vector
            z2: Second latent vector
            num_steps: Number of interpolation steps
            seq_len: Optional sequence length (defaults to max_seq_len)
            
        Returns:
            Interpolated motion sequences
        """
        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, num_steps, device=z1.device)
        
        # Process each interpolated vector individually and stack results
        results = []
        for alpha in alphas:
            # Interpolate between z1 and z2
            z_interp = (1-alpha)*z1 + alpha*z2
            # Decode the single interpolated vector
            result = self.decode(z_interp, seq_len)
            results.append(result)
        
        # Stack results along a new first dimension
        return torch.cat(results, dim=0)

    def compute_loss(self, x, reconstructed, mu, logvar, kld_weight=0, kl_weight_override=None):
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            x: Original input
            reconstructed: Reconstructed output
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            kld_weight: Weight for KL divergence term
            kl_weight_override: Override for KL weight
            
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss component
            kld_loss: KL divergence loss component
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence loss
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Apply weight override if provided
        effective_kld_weight = kl_weight_override if kl_weight_override is not None else kld_weight
        
        # Total loss
        total_loss = recon_loss + effective_kld_weight * kld_loss
        
        return total_loss, recon_loss, kld_loss

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    pose_dim = 126      # Dimension of pose parameters
    latent_dim = 256    # Latent dimension
    seq_len = 100       # Sequence length
    batch_size = 8      # Batch size
    
    # Create model instance
    model = ACTORVAE(
        pose_dim=pose_dim,
        latent_dim=latent_dim,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_seq_len=120
    ).to(device)
    print(f"Successfully created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, pose_dim).to(device)
    
    # Forward pass
    with torch.no_grad():
        reconstructed, mu, logvar = model(x)
    
    # Calculate loss
    total_loss, recon_loss, kld_loss = model.compute_loss(x, reconstructed, mu, logvar)
    
    # Print shapes and loss
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kld_loss.item():.4f}")
    
    # Test sampling
    samples = model.sample(num_samples=4, seq_len=seq_len)
    print(f"Sampled shape: {samples.shape}")
    
    # Test interpolation
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)
    interpolations = model.interpolate(z1, z2, num_steps=5, seq_len=seq_len)
    print(f"Interpolation shape: {interpolations.shape}")

if __name__ == "__main__":
    main()
