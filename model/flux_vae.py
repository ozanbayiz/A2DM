from dataclasses import dataclass

import torch
import math
from einops import rearrange
from torch import Tensor, nn


@dataclass
class MotionAutoEncoderParams:
    seq_length: int        # Time dimension (T) of input
    in_channels: int       # Input feature dimension (C)
    ch: int                # Base channel multiplier
    out_ch: int            # Output feature dimension
    ch_mult: list[int]     # Channel multipliers for each level
    num_res_blocks: int    # Number of residual blocks
    z_channels: int        # Latent dimension
    scale_factor: float    # Scaling factor for the latent
    shift_factor: float    # Shifting factor for the latent


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock1D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)

        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_.transpose(1, 2)).transpose(1, 2)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Shape: [batch, channels, seq_len]
        b, c, t = q.shape
        q = rearrange(q, "b c t -> b 1 t c").contiguous()
        k = rearrange(k, "b c t -> b 1 t c").contiguous()
        v = rearrange(v, "b c t -> b 1 t c").contiguous()
        
        # Apply scaled dot-product attention
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)
        
        # Reshape back to original format
        return rearrange(h_, "b 1 t c -> b c t", t=t, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.norm2 = nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample1D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # 1D convolution with stride 2 for downsampling
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        # Add padding on the right
        pad = (0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample1D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        # Use 1D interpolation 
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class MotionEncoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.seq_length = seq_length
        self.in_channels = in_channels
        
        # Initial projection from input channels to base channels
        self.conv_in = nn.Conv1d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_length = seq_length
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        
        # Downsampling blocks
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock1D(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            
            down = nn.Module()
            down.block = block
            down.attn = attn
            
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample1D(block_in)
                curr_length = curr_length // 2
            
            self.down.append(down)

        # Middle blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in, out_channels=block_in)

        # Output projection to 2*z_channels for mean and logvar
        # self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.norm_out = nn.LayerNorm(block_in, eps=1e-6, elementwise_affine=True)
        self.conv_out = nn.Conv1d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # x has shape [B, T, C] - reshape to [B, C, T]
        x = x.transpose(1, 2)
        
        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        
        # Output
        h = self.norm_out(h.transpose(1, 2)).transpose(1, 2)
        h = swish(h)
        h = self.conv_out(h)
        return h


class MotionDecoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        seq_length: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.seq_length = seq_length
        self.in_channels = in_channels
        self.down_factor = 2 ** (self.num_resolutions - 1)

        # Calculate the reduced sequence length
        min_seq_len = seq_length // self.down_factor
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.z_shape = (1, z_channels, min_seq_len)

        # Initial projection from latent to features
        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in, out_channels=block_in)

        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock1D(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            
            up = nn.Module()
            up.block = block
            up.attn = attn
            
            if i_level != 0:
                up.upsample = Upsample1D(block_in)
            
            self.up.insert(0, up)  # prepend to get consistent order

        # Output projection to the feature dimension
        # self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.norm_out = nn.LayerNorm(block_in, eps=1e-6, elementwise_affine=True)
        self.conv_out = nn.Conv1d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # Initial projection
        h = self.conv_in(z)

        # Middle blocks
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Final output
        h = self.norm_out(h.transpose(1, 2)).transpose(1, 2)
        h = swish(h)
        h = self.conv_out(h)
        
        # Output has shape [B, C, T] - reshape to [B, T, C]
        h = h.transpose(1, 2)
        
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class FluxVAE(nn.Module):
    def __init__(
        self,
        seq_length=120,       # Time dimension (T)
        in_channels=126,      # Feature dimension (C)
        ch=128,               # Base channel multiplier
        out_ch=126,           # Output feature dimension
        ch_mult=[1, 2, 4],    # Channel multipliers for each level
        num_res_blocks=2,     # Number of residual blocks
        z_channels=256,       # Latent dimension
        scale_factor=0.18215, # Scaling factor for the latent
        shift_factor=0.0,     # Shifting factor for the latent
    ):
        super().__init__()
        self.encoder = MotionEncoder(
            seq_length=seq_length,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )
        self.decoder = MotionDecoder(
            seq_length=seq_length,
            in_channels=in_channels,
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        mu, logvar = torch.chunk(z, 2, dim=1)
        return self.decode(z), mu, logvar
    def compute_loss(self, gt, reconstructed, mu, logvar, kld_weight=1.0, kl_weight_override=None):
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            gt: Original ground truth input
            reconstructed: Reconstructed output
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            kld_weight: Weight for KL divergence term
            kl_weight_override: Optional override for KL weight
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss component
            kld_loss: KL divergence loss component
        """
        # Use override weight if provided
        weight = kl_weight_override if kl_weight_override is not None else kld_weight
        
        # Reconstruction loss (MSE)
        recon_loss = torch.mean((gt - reconstructed) ** 2)
        
        # KL divergence loss
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + weight * kld_loss
        
        return total_loss, recon_loss, kld_loss


def main():
    """Test the forward pass of the Motion VAE model."""
    # Define model parameters
    seq_length = 120
    feature_dim = 126
    latent_dim = 256
    
    # Create model
    model = FluxVAE(
        seq_length=seq_length,
        in_channels=feature_dim,
        ch=128,
        out_ch=feature_dim,
        ch_mult=[1, 2, 4],  # Using three levels of downsampling
        num_res_blocks=2,
        z_channels=latent_dim,
        scale_factor=0.18215,
        shift_factor=0.0,
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create a random input tensor (B, T, C)
    batch_size = 8
    input_tensor = torch.randn(batch_size, seq_length, feature_dim)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Test encode
    with torch.no_grad():
        latent = model.encode(input_tensor)
        print(f"Latent tensor shape: {latent.shape}")
        
        # Test decode
        output = model.decode(latent)
        print(f"Output tensor shape: {output.shape}")
        
        # Test full forward pass
        reconstructed, mu, logvar = model(input_tensor)
        print(f"Reconstructed tensor shape: {reconstructed.shape}")
        print(f"Mu tensor shape: {mu.shape}")
        print(f"Logvar tensor shape: {logvar.shape}")
        
        # Calculate reconstruction error
        loss, recon_loss, kl_loss = model.compute_loss(input_tensor, reconstructed, mu, logvar)
        print(f"Total loss: {loss.item():.6f}")
        print(f"Reconstruction loss: {recon_loss.item():.6f}")
        print(f"KL loss: {kl_loss.item():.6f}")


if __name__ == "__main__":
    main()