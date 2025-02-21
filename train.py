import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.simple_vae import VAE
from models.resnet_vae import ResNetVAE
from dataloader import get_dataloader
from tqdm import tqdm

def loss_function(recon_x, x, mu, logvar, kl_loss_weight=0.01, mask=None):
    """
    VAE loss: Reconstruction (MSE) loss + KL divergence.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='none') * mask
    recon_loss = recon_loss.sum()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(f"recon_loss: {recon_loss:.4f}, kl_loss: {kl_loss*kl_loss_weight:.4f}")
    return recon_loss + kl_loss * kl_loss_weight

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0
    # for batch_idx, (data, mask) in enumerate(tqdm(dataloader, desc="Training")):
    for batch_idx, (data, mask) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        mask = mask.to(device)
        loss = loss_function(recon, data, mu, logvar, kl_loss_weight=0.1, mask=mask)
        loss.backward()
        train_loss += loss.item()   
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss per sample = {loss.item() / data.size(0):.4f}")
    avg_loss = train_loss / len(dataloader.dataset)
    print(f"====> Average training loss: {avg_loss:.4f}")
    return avg_loss

def main():
    # --- Hyperparameters ---
    data_dir = "./sns_slahmr"  # Change this to your data directory
    batch_size = 32
    epochs = 50
    latent_dim = 128
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    dataloader = get_dataloader(data_dir, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # model = VAE(latent_dim=latent_dim).to(device)
    model = ResNetVAE(
        latent_dim=latent_dim,
        T_in=100,
        input_dim=138,
        encoder_output_channels=512,
        down_t=4,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=1,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create a directory for checkpoints if it doesn't exist.
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        train_loss = train_epoch(model, dataloader, optimizer, device)
        # Save checkpoint every epoch.
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"vae_epoch_{epoch}.pt"))

if __name__ == "__main__":
    main()
