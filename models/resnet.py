import torch
import torch.nn as nn

class Nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=0.2):
        super(ResConv1DBlock, self).__init__()
        padding = dilation

        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = Nonlinearity()
            self.activation2 = Nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        else:
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        self.conv1 = nn.Conv1d(n_in, n_state, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_orig = x
        # Apply normalization and activation.
        # For LayerNorm we need to transpose since it expects (..., features)
        if isinstance(self.norm1, nn.LayerNorm):
            x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        if isinstance(self.norm2, nn.LayerNorm):
            x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            x = self.norm2(x)
        x = self.activation2(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x + x_orig

class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        blocks = [
            ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
