import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import argparse
import yaml
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
    return recon_loss + kl_loss * kl_loss_weight

def train_epoch(model, dataloader, optimizer, device, epoch, rank):
    model.train()
    train_loss = 0
    # Ensure the DistributedSampler shuffles differently each epoch.
    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)
    for batch_idx, (data, mask) in enumerate(tqdm(dataloader, desc="Training", disable=(rank != 0))):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        mask = mask.to(device)
        loss = loss_function(recon, data, mu, logvar, kl_loss_weight=0.1, mask=mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(dataloader.dataset)
    if rank == 0:
        print(f"====> Average training loss: {avg_loss:.4f}")
    return avg_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    vae_conf = config['VAE']
    train_conf = config['Training']

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
        dataset = get_dataloader(train_conf['data_dir'], batch_size=train_conf['batch_size'], shuffle=True).dataset
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = get_dataloader(train_conf['data_dir'], batch_size=train_conf['batch_size'], shuffle=True, num_workers=4, sampler=sampler)

    # Build model.
    model = ResNetVAE(**vae_conf).to(device)
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    optimizer = optim.Adam(model.parameters(), lr=train_conf['learning_rate'])

    # Create a directory for checkpoints only on the master process.
    checkpoint_dir = train_conf.get('checkpoint_dir', 'checkpoints')
    if args.local_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(1, train_conf['epochs'] + 1):
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
