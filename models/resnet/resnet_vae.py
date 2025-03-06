import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from encdec import Encoder, Decoder

class ResNetVAE(nn.Module):
    def __init__(
        self, 
        latent_dim: int=128, 
        T_in: int=100, 
        input_dim: int=138,
        encoder_output_channels: int=512, 
        down_t: int=2, # number of downsampling blocks
        stride_t: int=2, # stride of the downsampling blocks
        width: int=512,
        depth: int=3,
        dilation_growth_rate: int=3,
        pos_emb_dim: int=0,
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
        self.pos_emb_dim = pos_emb_dim

        # Build encoder and decoder.
        self.encoder = Encoder(
            T_in=T_in,
            input_emb_width=input_dim + pos_emb_dim*4,
            enc_emb_width=encoder_output_channels,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
        )
        # Calculate T_out after downsampling.
        self.T_out = T_in // (stride_t ** down_t)
        self.flattened_size = encoder_output_channels * self.T_out

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = Decoder(
            T_out=T_in,
            enc_emb_width=encoder_output_channels,
            output_emb_width=input_dim,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
        )

    @staticmethod
    def encode_sine_cosine(x: torch.Tensor, emb_dim: int=78) -> torch.Tensor:
        """
        Encodes a tensor of shape (B, T, 3) into a higher-dimensional tensor using sine and cosine embeddings.
        
        For each of the 3 coordinates, we apply K frequency functions to produce
        [sin(freq * x), cos(freq * x)] pairs, resulting in an embedding of size 6*K.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, 3).
            emb_dim (int): Desired output embedding dimension, must be a multiple of 6.
        
        Returns:
            torch.Tensor: Output tensor of shape (B, T, emb_dim).
        """
        # Ensure that emb_dim is divisible by 6.
        assert emb_dim % 6 == 0, "Embedding dimension must be a multiple of 6."
        
        # Determine the number of frequencies per coordinate.
        num_freqs = emb_dim // 6
        
        # Create a tensor of frequencies. Here we use powers of 2 multiplied by pi.
        # This means the i-th frequency is: 2**i * pi for i in range(num_freqs).
        freqs = (2 ** torch.arange(num_freqs, dtype=x.dtype, device=x.device)) * math.pi
        
        # Reshape x to (B, T, 3, 1) so that we can broadcast multiplication with freqs (shape: (num_freqs,))
        x_expanded = x.unsqueeze(-1)  # shape: (B, T, 3, 1)
        
        # Multiply each coordinate by all frequency values.
        x_scaled = x_expanded * freqs  # shape: (B, T, 3, num_freqs)
        
        # Compute sine and cosine embeddings.
        sin_embed = torch.sin(x_scaled)  # shape: (B, T, 3, num_freqs)
        cos_embed = torch.cos(x_scaled)  # shape: (B, T, 3, num_freqs)
        
        # Concatenate sine and cosine along the last dimension to get pairs.
        # The resulting shape is (B, T, 3, 2*num_freqs).
        embed = torch.cat([sin_embed, cos_embed], dim=-1)
        
        # Flatten the last two dimensions so that the final shape is (B, T, 3 * 2 * num_freqs) == (B, T, emb_dim)
        embed = embed.view(x.shape[0], x.shape[1], -1)
    
        return embed
    

    def encode(self, x):
        """
        x: (B, T, input_dim)
        Returns:
            mu, logvar: each of shape (B, latent_dim)
        """
        if self.pos_emb_dim > 0:
            person_1_trans = x[:, :, :3]
            person_2_trans = x[:, :, 69:72]
            person_1_trans = self.encode_sine_cosine(person_1_trans)
            person_2_trans = self.encode_sine_cosine(person_2_trans)

            person_1_orient = x[:, :, 3:6]
            person_2_orient = x[:, :, 72:75]
            person_1_orient = self.encode_sine_cosine(person_1_orient)
            person_2_orient = self.encode_sine_cosine(person_2_orient)

            x = torch.cat([x, person_1_trans, person_2_trans, person_1_orient, person_2_orient], dim=-1)
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

# For testing purposes:
if __name__ == '__main__':
    model = ResNetVAE(latent_dim=128, T_in=100, input_dim=138)
    x = torch.randn(8, 100, 138)  # Batch of 8 sequences
    recon, mu, logvar = model(x)
    print("Reconstruction shape:", recon.shape)  # Expected: (8, 100, 138)
    print("Mu shape:", mu.shape)                # Expected: (8, 128)
    print("Logvar shape:", logvar.shape)        # Expected: (8, 128)
