import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import argparse
from models.simple_vae import VAE
from models.resnet_vae import ResNetVAE
from dataloader import get_dataloader
from tqdm import tqdm

def loss_function(recon_x, x, mu, logvar, kl_loss_weight=0.01):
    """
    VAE loss: Reconstruction loss (L1 or MSE) + KL divergence.
    """
    recon_loss = F.l1_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss * kl_loss_weight

def train_epoch(model, dataloader, optimizer, device, epoch, rank):
    model.train()
    train_loss = 0
    # Ensure the DistributedSampler shuffles differently each epoch.
    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)
    for batch_idx, data in enumerate(tqdm(dataloader, desc="Training", disable=(rank != 0))):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar, kl_loss_weight=0.1)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(dataloader.dataset)
    if rank == 0:
        print(f"====> Average training loss: {avg_loss:.4f}")
    return avg_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./sns_slahmr", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()

    # Check if distributed training is enabled.
    distributed = False
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        distributed = True
        dist.init_process_group(backend='nccl')
    
    # Set the proper CUDA device.
    if distributed:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloader and (if distributed) a DistributedSampler.
    if distributed:
        # Create the dataset first so we can wrap it in a DistributedSampler.
        dataset = get_dataloader(args.data_dir, batch_size=args.batch_size, shuffle=True).dataset
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size, shuffle=True, num_workers=4, sampler=sampler)

    # Build model.
    model = ResNetVAE(
        latent_dim=args.latent_dim,
        T_in=100,
        input_dim=138,
        encoder_output_channels=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm='LN'
    ).to(device)
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Create a directory for checkpoints only on the master process.
    checkpoint_dir = "checkpoints"
    if args.local_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        if distributed:
            sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print(f"Epoch {epoch}")
        train_loss = train_epoch(model, dataloader, optimizer, device, epoch, args.local_rank)
        if args.local_rank == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"vae_epoch_{epoch}.pt"))
    
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
