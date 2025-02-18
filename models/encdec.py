import torch.nn as nn
from resnet import Resnet1D

class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width=138,      # for our duet motion (138 features)
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm='BN'
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
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        # Final convolution to produce output embedding.
        blocks.append(nn.Conv1d(width, output_emb_width, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        # Expect input x shape: (B, T, input_emb_width)
        # Permute to (B, input_emb_width, T) for Conv1d.
        x = x.permute(0, 2, 1)
        return self.model(x)  # Output shape: (B, output_emb_width, T_out)

class Decoder(nn.Module):
    def __init__(self,
                 output_emb_width=512,
                 input_emb_width=138,  # final desired output channels
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm='BN'):
        super().__init__()
        blocks = []
        # Initial convolution to expand latent features.
        blocks.append(nn.Conv1d(output_emb_width, width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.ReLU())
        # Upsample in time.
        for _ in range(down_t):
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=stride_t, mode='nearest'),
                nn.Conv1d(width, width, kernel_size=3, stride=1, padding=1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        # x is expected to be shape: (B, output_emb_width, T_out)
        x = self.model(x)
        # Permute back to (B, T, input_emb_width)
        return x.permute(0, 2, 1)
