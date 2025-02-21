import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.simple_vae import VAE
from models.resnet_vae import ResNetVAE
from dataloader import get_dataloader
from tqdm import tqdm
import yaml
import argparse
def loss_function(recon_x, x, mu, logvar, kl_loss_weight=0.01, mask=None):
    """
    VAE loss: Reconstruction (MSE) loss + KL divergence.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='none') * mask
    recon_loss = recon_loss.sum()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(f"recon_loss: {recon_loss:.4f}, kl_loss: {kl_loss*kl_loss_weight:.4f}")
    return recon_loss + kl_loss * kl_loss_weight

def train_epoch(model, dataloader, optimizer, device, kl_loss_weight):
    model.train()
    train_loss = 0
    # for batch_idx, (data, mask) in enumerate(tqdm(dataloader, desc="Training")):
    for batch_idx, (data, mask) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        mask = mask.to(device)
        loss = loss_function(recon, data, mu, logvar, kl_loss_weight=kl_loss_weight, mask=mask)
        loss.backward()
        train_loss += loss.item()   
        optimizer.step()

        # if batch_idx % 10 == 0:
        #     print(f"Batch {batch_idx}: Loss per sample = {loss.item() / data.size(0):.4f}")
    avg_loss = train_loss / len(dataloader.dataset)
    print(f"====> Average training loss: {avg_loss:.4f}")
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("--config", type=str, default="configs/vae_config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    vae_config = config['VAE']
    train_config = config['Training']

    # --- Hyperparameters from config ---
    data_dir = "./sns_slahmr_64"  # Change this to your data directory
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    kl_loss_weight = train_config['kl_loss_weight']
    device = torch.device(train_config['device'])

    dataloader = get_dataloader(data_dir, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # model = ResNetVAE(
    #     latent_dim=vae_config['latent_dim'],
    #     T_in=vae_config['T_in'],
    #     input_dim=vae_config['input_dim'],
    #     encoder_output_channels=vae_config['encoder_output_channels'],
    #     down_t=vae_config['down_t'],
    #     stride_t=vae_config['stride_t'],
    #     width=vae_config['width'],
    #     depth=vae_config['depth'],
    #     dilation_growth_rate=vae_config['dilation_growth_rate'],
    # ).to(device)
    model = ResNetVAE(**vae_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create a directory for checkpoints if it doesn't exist.
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        train_loss = train_epoch(model, dataloader, optimizer, device, kl_loss_weight)
        # Save checkpoint every epoch.
        torch.save(model.cpu().state_dict(), os.path.join(checkpoint_dir, f"vae_epoch_{epoch}.pt"))
        model.to(device)

if __name__ == "__main__":
    main()
