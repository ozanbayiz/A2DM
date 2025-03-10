import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock1D(nn.Module):
    """1D ResNet block with GroupNorm and SiLU activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=8):
        super().__init__()
        self.use_conv_shortcut = in_channels != out_channels
        
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        
        if self.use_conv_shortcut:
            self.conv_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        
        return x + h


class TemporalAttention(nn.Module):
    """Temporal attention using nn.MultiheadAttention"""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=False
        )
        
    def forward(self, x):
        # x: [B, C, T]
        # Convert to [T, B, C] format for MultiheadAttention
        x_attn = x.permute(2, 0, 1)
        
        # Apply self-attention
        attn_output, _ = self.attention(x_attn, x_attn, x_attn)
        
        # Convert back to [B, C, T]
        output = attn_output.permute(1, 2, 0)
        
        return output


class TemporalDownsample(nn.Module):
    """Temporal downsampling using strided 1D convolution"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=4, stride=2, padding=1
        )
    
    def forward(self, x):
        return self.conv(x)


class TemporalUpsample(nn.Module):
    """Temporal upsampling using transposed 1D convolution"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1
        )
    
    def forward(self, x):
        return self.conv(x)


class VideoVAEPlus(nn.Module):
    """VAE model for video data with 1D convolutions and temporal attention"""
    
    def __init__(
        self,
        input_dim,  # Flattened dimension (C*H*W)
        base_channels=256,  # Base channel count
        channel_multipliers=(1, 2, 4, 8),  # Channel multipliers at each level
        num_res_blocks=2,  # Number of ResNet blocks per level
        z_dim=256,  # Latent dimension
        use_attention=True,  # Whether to use attention
        attention_heads=8,  # Number of attention heads
        norm_groups=32,  # Number of groups for GroupNorm
    ):
        super().__init__()
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.z_dim = z_dim
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.norm_groups = norm_groups
        
        # Calculate channels at each resolution level
        self.channels = [base_channels * m for m in channel_multipliers]
        
        # Encoder components
        self.encoder_input_conv = nn.Conv1d(self.input_dim, base_channels, kernel_size=3, padding=1)
        
        # Encoder downsampling path
        self.encoder_down_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        self.encoder_downsamplers = nn.ModuleList()
        
        in_ch = base_channels
        for i, ch in enumerate(self.channels):
            # Add ResNet blocks
            res_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                res_blocks.append(ResNetBlock1D(in_ch, ch, groups=norm_groups))
                in_ch = ch
            self.encoder_down_blocks.append(res_blocks)
            
            # Add attention if needed
            if use_attention:
                self.encoder_attentions.append(TemporalAttention(ch, num_heads=attention_heads))
            else:
                self.encoder_attentions.append(nn.Identity())
            
            # Add downsampler if not last level
            if i < len(self.channels) - 1:
                self.encoder_downsamplers.append(TemporalDownsample(ch, ch))
            else:
                self.encoder_downsamplers.append(nn.Identity())
        
        # Middle blocks
        self.mid_block1 = ResNetBlock1D(self.channels[-1], self.channels[-1], groups=norm_groups)
        self.mid_attn = TemporalAttention(self.channels[-1], num_heads=attention_heads) if use_attention else nn.Identity()
        self.mid_block2 = ResNetBlock1D(self.channels[-1], self.channels[-1], groups=norm_groups)
        
        # To latent space
        self.to_latent = nn.Sequential(
            nn.GroupNorm(norm_groups, self.channels[-1]),
            nn.SiLU(),
            nn.Conv1d(self.channels[-1], 2 * z_dim, kernel_size=1)
        )
        
        # From latent space
        self.from_latent = nn.Conv1d(z_dim, self.channels[-1], kernel_size=1)
        
        # Decoder components
        self.decoder_up_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        self.decoder_upsamplers = nn.ModuleList()
        
        reversed_channels = list(reversed(self.channels))
        
        for i, ch in enumerate(reversed_channels):
            # Previous channels (for skips if implemented)
            out_ch = reversed_channels[i+1] if i < len(reversed_channels) - 1 else base_channels
            
            # Add ResNet blocks
            res_blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):  # Extra block in decoder for better results
                res_ch = ch if j == 0 else out_ch
                res_blocks.append(ResNetBlock1D(res_ch, out_ch, groups=norm_groups))
            self.decoder_up_blocks.append(res_blocks)
            
            # Add attention if needed
            if use_attention:
                self.decoder_attentions.append(TemporalAttention(out_ch, num_heads=attention_heads))
            else:
                self.decoder_attentions.append(nn.Identity())
            
            # Add upsampler if not last level
            if i < len(reversed_channels) - 1:
                self.decoder_upsamplers.append(TemporalUpsample(out_ch, out_ch))
            else:
                self.decoder_upsamplers.append(nn.Identity())
        
        # Final output conversion
        self.final_conv = nn.Sequential(
            nn.GroupNorm(norm_groups, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, self.input_dim, kernel_size=3, padding=1)
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        # x: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        
        # Initial feature extraction
        h = self.encoder_input_conv(x)
        
        # Downsampling path
        for i, (res_blocks, attn, downsample) in enumerate(zip(
            self.encoder_down_blocks, self.encoder_attentions, self.encoder_downsamplers
        )):
            # Apply ResNet blocks
            for res_block in res_blocks:
                h = res_block(h)
            
            # Apply attention
            h = attn(h)
            
            # Apply downsampling
            h = downsample(h)
        
        # Middle blocks
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # To latent parameters
        latent_dist = self.to_latent(h)
        
        # Split into mean and log variance
        mu, log_var = latent_dist.chunk(2, dim=1)
        
        return mu, log_var
    
    def sample_latent(self, mu, log_var):
        """Sample from the latent distribution using reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent representation back to input space"""
        # z: [B, z_dim, T_encoded]
        h = self.from_latent(z)
        
        # Middle blocks
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Upsampling path
        for i, (res_blocks, attn, upsample) in enumerate(zip(
            self.decoder_up_blocks, self.decoder_attentions, self.decoder_upsamplers
        )):
            # Apply ResNet blocks
            for res_block in res_blocks:
                h = res_block(h)
            
            # Apply attention
            h = attn(h)
            
            # Apply upsampling
            h = upsample(h)
        
        # Final output conversion
        x_recon = self.final_conv(h)
        
        # x_recon: [B, D, T] -> [B, T, D]
        x_recon = x_recon.transpose(1, 2)
        
        return x_recon
    
    def forward(self, x):
        """Full forward pass through the VAE"""
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample latent
        z = self.sample_latent(mu, log_var)
        
        # Decode
        x_recon = self.decode(z)
        
        return x_recon, mu, log_var
    
    def get_kl_loss(self, mu, log_var):
        """Compute KL divergence loss"""
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return kl_loss.mean()

    @staticmethod
    def reconstruction_loss(x_recon, x_target):
        """Compute reconstruction loss (MSE)"""
        return F.mse_loss(x_recon, x_target)
    
    def loss_function(self, x_recon, x_target, mu, log_var, kl_weight=1.0):
        """Compute combined VAE loss (reconstruction + KL)"""
        recon_loss = self.reconstruction_loss(x_recon, x_target)
        kl_loss = self.get_kl_loss(mu, log_var)
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


def main():
    """Test the VideoVAEPlus model with a dummy input"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Model parameters
    input_dim = 63  # Example: 4 (RGBA) * 16 * 16 (image size)
    batch_size = 2
    seq_length = 100
    
    # Initialize model
    model = VideoVAEPlus(
        input_dim=input_dim,
        base_channels=128,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        z_dim=256,
        use_attention=True,
        attention_heads=8,
        norm_groups=16
    )
    
    # Create dummy input: [B, T, D]
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    x_recon, mu, log_var = model(x)
    
    # Calculate loss
    loss, recon_loss, kl_loss = model.loss_function(x_recon, x, mu, log_var, kl_weight=0.1)
    
    # Print shapes and loss
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log_var shape: {log_var.shape}")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Check if shapes match
    assert x.shape == x_recon.shape, "Input and reconstruction shapes don't match!"
    print("Test passed: Input and reconstruction shapes match.")


if __name__ == "__main__":
    main()
