import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create standard positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, seq_len=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            seq_len: Optional override for sequence length
        """
        if seq_len is None:
            seq_len = x.size(1)
        return self.pe[:, :seq_len]

class ACTORVAE(nn.Module):
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
        
        # Input projection: from pose_dim to hidden_dim
        self.input_projection = nn.Linear(pose_dim, hidden_dim)
        
        # Special tokens for mean and log-variance
        self.mu_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.logvar_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Positional encoding for sequence
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Encoder transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Projection from hidden dim to latent space
        self.mu_projection = nn.Linear(hidden_dim, latent_dim)
        self.logvar_projection = nn.Linear(hidden_dim, latent_dim)
        
        # Latent to hidden projection for decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        
        # Query embeddings for decoder (learnable per time step)
        self.query_embeddings = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        
        # Decoder transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection from hidden to pose
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pose_dim)
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
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # [B, T, H]
        
        # Add positional encoding
        pos_encoding = self.pos_encoder(x)  # [1, T, H]
        x = x + pos_encoding  # [B, T, H]
        
        # Prepend special tokens for mu and logvar
        mu_tokens = self.mu_token.expand(batch_size, -1, -1)  # [B, 1, H]
        logvar_tokens = self.logvar_token.expand(batch_size, -1, -1)  # [B, 1, H]
        
        # Concatenate special tokens with input
        x_with_special = torch.cat([mu_tokens, logvar_tokens, x], dim=1)  # [B, 2+T, H]
        
        # Pass through transformer encoder
        encoder_output = self.transformer_encoder(x_with_special)  # [B, 2+T, H]
        
        # Extract and project special token outputs
        mu_hidden = encoder_output[:, 0]  # [B, H]
        logvar_hidden = encoder_output[:, 1]  # [B, H]
        
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
        
        # Project latent to hidden dimension and repeat for memory
        z_hidden = self.latent_to_hidden(z)  # [B, H]
        memory = z_hidden.unsqueeze(1)  # [B, 1, H]
        
        # Get query embeddings for desired sequence length
        query_emb = self.query_embeddings[:, :seq_len, :]  # [1, seq_len, H]
        
        # Add positional encoding to query embeddings
        pos_encoding = self.pos_encoder(query_emb, seq_len)  # [1, seq_len, H]
        query_emb = query_emb + pos_encoding  # [1, seq_len, H]
        
        # Expand queries to batch size
        query_emb = query_emb.expand(batch_size, -1, -1)  # [B, seq_len, H]
        
        # Pass through transformer decoder
        # No need for attention mask since we're doing non-autoregressive decoding
        decoder_output = self.transformer_decoder(
            tgt=query_emb,  # [B, seq_len, H]
            memory=memory,  # [B, 1, H]
        )  # [B, seq_len, H]
        
        # Project back to pose space
        output = self.output_projection(decoder_output)  # [B, seq_len, pose_dim]
        
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
            z: Sampled latent vector
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
            
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss component
            kld_loss: KL divergence loss component
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence loss
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + kld_weight * kld_loss
        
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
