import torch
import torch.nn as nn

class ResConv1DBlock(nn.Module):
    def __init__(
        self, 
        n_in, 
        n_state,
        dilation=1, 
        dropout=0.1
    ):
        super().__init__()

        padding = dilation

        self.norm1 = nn.LayerNorm(n_in)
        self.norm2 = nn.LayerNorm(n_in)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.conv1 = nn.Conv1d(n_in, n_state, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_orig = x

        x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.act2(x)
        x = self.conv2(x)

        x = self.dropout(x)

        return x + x_orig

class Resnet1D(nn.Module):
    def __init__(
        self, 
        n_in: int, 
        n_depth: int, 
        dilation_growth_rate: int=1, 
        reverse_dilation: bool=False, 
    ):
        super().__init__()
        blocks = [
            ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth)
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
