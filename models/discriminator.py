import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, channels, kernel_size):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, n_layers):
        super(Discriminator, self).__init__()

        self.in_conv = nn.Conv1d(in_channels, channels, 1)
        self.act = nn.SiLU()
        self.layers = nn.ModuleList([
            ConvLayer(
                channels,
                kernel_size
            ) for _ in range(n_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_conv = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.act(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.pool(x)
        x = self.out_conv(x)
        return x
