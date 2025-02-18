import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(
        self, 
        in_dim=138,
        hidden_dim=256,
        latent_dim=128
    ):
        super().__init__()

        # Encoder: expects input shape (B, in_dim, 100)
        self.encoder = nn.Sequential(
            # Input: (B, in_dim, 100)
            nn.Conv1d(in_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # -> (B, hidden_dim, 50)
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1),  # -> (B, hidden_dim*2, 25)
            nn.ReLU(),
            nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=4, stride=2, padding=1),  # -> (B, hidden_dim*2, 12)
            nn.ReLU(),
            nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=4, stride=2, padding=1),  # -> (B, hidden_dim*2, 6)
            nn.ReLU(),
        )
        self.flattened_size = hidden_dim * 2 * 6

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Decoder: reconstructs to (B, in_dim, 100)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            # Start from shape (B, hidden_dim*2, 6)
            nn.ConvTranspose1d(hidden_dim*2, hidden_dim*2, kernel_size=4, stride=2, padding=1),  # -> (B, hidden_dim*2, 12)
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding=1),  # -> (B, hidden_dim, 24)
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1),  # -> (B, hidden_dim//2, 48)
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim//2, in_dim, kernel_size=4, stride=2, padding=1),  # -> (B, in_dim, 96)
            nn.ReLU(),
            # Since our target length is 100, we use an interpolation layer:
            nn.Upsample(size=100, mode='linear', align_corners=False),
        )

    def encode(self, x):
        # x: (B, 100, 138) -> rearrange to (B, 138, 100)
        x = x.permute(0, 2, 1)
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(z.size(0), 512, 6)
        x_recon = self.decoder(h)
        # Rearrange back to (B, 100, 138)
        x_recon = x_recon.permute(0, 2, 1)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

if __name__ == '__main__':
    # Quick test of the model dimensions
    model = VAE(latent_dim=128)
    x = torch.randn(8, 100, 138)  # Batch of 8 samples
    recon, mu, logvar = model(x)
    print("Reconstruction shape:", recon.shape)  # Expected: (8, 100, 138)
    print("Mu shape:", mu.shape)                # Expected: (8, 128)
    print("Logvar shape:", logvar.shape)        # Expected: (8, 128)
