import torch.nn as nn
from .resnet import Resnet1D

class Encoder(nn.Module):
    def __init__(
        self,
        T_in: int, # number of time frames in input
        input_emb_width: int, # number of features per frame
        enc_emb_width: int, # number of channels output by the encoder
        down_t: int, # number of downsampling blocks
        stride_t: int, # stride of the downsampling blocks
        width: int, # width of the encoder
        depth: int, # depth of the encoder
        dilation_growth_rate: int, # dilation growth rate
    ):
        super().__init__()
        blocks = []
        # Initial convolution: (B, input_emb_width, T) -> (B, width, T)
        blocks.append(nn.Conv1d(input_emb_width, width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.ReLU())
        # Downsample in time with a series of blocks.
        filter_t, pad_t = stride_t * 2, stride_t // 2
        for _ in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, kernel_size=filter_t, stride=stride_t, padding=pad_t),
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=False),
            )
            blocks.append(block)
        # Final convolution to produce output embedding.
        blocks.append(nn.Conv1d(width, enc_emb_width, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        # Expect input x shape: (B, T, input_emb_width)
        # Permute to (B, input_emb_width, T) for Conv1d.
        x = x.permute(0, 2, 1)
        return self.model(x)  # Output shape: (B, enc_emb_width, T_in)

class Decoder(nn.Module):
    def __init__(
        self,
        T_out: int, # number of time frames in output
        enc_emb_width: int, # number of channels output by the encoder
        output_emb_width: int, # number of features per frame
        down_t: int, # number of downsampling blocks
        stride_t: int, # stride of the downsampling blocks
        width: int, # width of the decoder
        depth: int, # depth of the decoder
        dilation_growth_rate: int, # dilation growth rate
    ):
        super().__init__()
        blocks = []
        # Initial convolution to expand latent features.
        blocks.append(nn.Conv1d(enc_emb_width, width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.ReLU())
        # Upsample in time.
        for _ in range(down_t):
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, width, kernel_size=3, stride=1, padding=1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, output_emb_width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.Upsample(size=T_out, mode='linear', align_corners=False))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        # x is expected to be shape: (B, enc_emb_width, T_in)
        x = self.model(x)
        # Permute back to (B, T, output_emb_width)
        return x.permute(0, 2, 1)
