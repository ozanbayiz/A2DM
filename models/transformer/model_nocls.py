import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import TransformerWrapper, Encoder  # ensure x-transformers is installed

class MotionVAE(nn.Module):
    def __init__(
        self,
        input_dim,          # feature dimension per timestep (e.g., 69)
        latent_dim,         # size of latent vector (controllable)
        timesteps,          # number of timesteps (e.g., 100)
        encoder_depth=4,
        decoder_depth=4,
        heads=8,
        mlp_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.timesteps = timesteps

        # --- Encoder ---
        # Instead of prepending a CLS token, we directly encode the sequence and then aggregate
        self.encoder = Encoder(
            dim=input_dim,
            depth=encoder_depth,
            heads=heads,
            ff_mult=mlp_dim // input_dim,  # ff_mult * dim = mlp_dim
            attn_dropout=dropout,
            ff_dropout=dropout,
            rotary_pos_emb=True
        )
        # Aggregate the temporal dimension via mean pooling, then project to latent params.
        self.to_latent_params = nn.Linear(input_dim, latent_dim * 2)

        # --- Decoder ---
        # The decoder uses cross-attention: a set of learned query tokens decode using context from the latent vector.
        self.decoder = Encoder(
            dim=input_dim,
            depth=decoder_depth,
            heads=heads,
            ff_mult=mlp_dim // input_dim,
            attn_dropout=dropout,
            ff_dropout=dropout,
            cross_attend=True,
            rotary_pos_emb=True
        )
        # Learned query tokens (one per timestep)
        self.query_tokens = nn.Parameter(torch.randn(1, timesteps, input_dim))
        # Project the latent vector to a context vector for the decoder.
        self.latent_to_context = nn.Linear(latent_dim, input_dim)
        # Final projection back to the output space.
        self.to_output = nn.Linear(input_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        x: motion tensor of shape (B, T, input_dim)
        Returns:
          out: reconstructed motion of shape (B, T, input_dim)
          mu, logvar: latent parameters for KL divergence loss.
        """
        B, T, D = x.shape
        assert T == self.timesteps, "Input sequence length must match specified timesteps."

        # --- Encoding ---
        # Encode the sequence; output shape: (B, T, D)
        encoded = self.encoder(x)
        # Aggregate across the time dimension (mean pooling) to obtain a global representation.
        pooled = encoded.mean(dim=1)  # (B, D)
        latent_params = self.to_latent_params(pooled)  # (B, 2*latent_dim)
        mu, logvar = latent_params.chunk(2, dim=-1)     # each: (B, latent_dim)
        z = self.reparameterize(mu, logvar)              # (B, latent_dim)

        # --- Decoding ---
        # Prepare learned query tokens for decoding.
        queries = self.query_tokens.expand(B, -1, -1)    # (B, T, D)
        # Convert the latent vector to a context vector.
        context = self.latent_to_context(z)              # (B, D)
        # Expand context along the temporal dimension.
        context = context.unsqueeze(1).expand(B, self.timesteps, D)  # (B, T, D)
        # Decode using cross-attention: query tokens attend to the latent-derived context.
        decoded = self.decoder(queries, context=context) # (B, T, D)
        out = self.to_output(decoded)                    # (B, T, D)

        return out, mu, logvar

# --- Example Usage ---
if __name__ == '__main__':
    # Suppose our human motion data has 100 timesteps and 69 features.
    B, T, D = 8, 100, 69
    latent_dim = 32  # control the latent dimension here
    model = MotionVAE(input_dim=D, latent_dim=latent_dim, timesteps=T)

    # Create dummy input data
    x = torch.randn(B, T, D)
    recon, mu, logvar = model(x)
    print("Reconstructed shape:", recon.shape)  # Expected: (B, T, D)
