import torch
import torch.nn as nn
import torch.nn.functional as F
from encdec import Encoder, Decoder

class ResNetVAE(nn.Module):
    def __init__(
            self, 
            latent_dim=128, 
            T_in=100, 
            input_dim=138,
            encoder_output_channels=512, 
            down_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation='relu',
            norm='BN'
        ):
        """
        Args:
            latent_dim: Dimensionality of the latent space.
            T_in: Number of time frames in input (e.g., 100).
            input_dim: Number of features per frame (e.g., 138).
            encoder_output_channels: Number of channels output by the encoder.
            down_t: Number of downsampling blocks.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.T_in = T_in

        # Build encoder and decoder.
        self.encoder = Encoder(
            input_emb_width=input_dim,
            output_emb_width=encoder_output_channels,
            down_t=down_t,
            stride_t=2,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm
        )
        # Calculate T_out after downsampling.
        # With each block downsampling by a factor of 2, T_out = T_in / (2^down_t).
        self.T_out = T_in // (2 ** down_t)
        self.flattened_size = encoder_output_channels * self.T_out

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = Decoder(
            output_emb_width=encoder_output_channels,
            input_emb_width=input_dim,
            down_t=down_t,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation='relu',
            norm='BN'
        )

    def encode(self, x):
        """
        x: (B, T, input_dim)
        Returns:
            mu, logvar: each of shape (B, latent_dim)
        """
        # Permute input to (B, input_dim, T) for Conv1d.
        h = self.encoder(x)  # -> (B, encoder_output_channels, T_out)
        h = h.view(x.size(0), -1)  # Flatten to (B, flattened_size)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(z.size(0), -1, self.T_out)  # -> (B, encoder_output_channels, T_out)
        x_recon = self.decoder(h)  # -> (B, T, input_dim)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """
    VAE loss = reconstruction loss (MSE) + KL divergence.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# For testing purposes:
if __name__ == '__main__':
    model = ResNetVAE(latent_dim=128, T_in=100, input_dim=138)
    x = torch.randn(8, 100, 138)  # Batch of 8 sequences
    recon, mu, logvar = model(x)
    print("Reconstruction shape:", recon.shape)  # Expected: (8, 100, 138)
    print("Mu shape:", mu.shape)                # Expected: (8, 128)
    print("Logvar shape:", logvar.shape)        # Expected: (8, 128)
