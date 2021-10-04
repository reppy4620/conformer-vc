import torch.fft
import torch.nn as nn

from .attention import RelativeSelfAttentionLayer
from .common import LayerNorm, FFN


class ConformerLayer(nn.Module):
    def __init__(self,
                 channels,
                 n_heads,
                 dropout):
        super(ConformerLayer, self).__init__()

        self.ff1 = FFN(channels, dropout)
        self.mha = RelativeSelfAttentionLayer(channels, n_heads, dropout)

        self.ff2 = FFN(channels, dropout)

        self.norm_post = LayerNorm(channels)

    def forward(self, x, pos_emb, x_mask):
        x += 0.5 * self.ff1(x, x_mask)
        x += self.mha(x, pos_emb, x_mask)
        x += torch.real(torch.fft.fft2(x))
        x += 0.5 * self.ff2(x, x_mask)
        x = self.norm_post(x)
        x *= x_mask
        return x


class Conformer(nn.Module):
    def __init__(self,
                 channels=384,
                 n_layers=4,
                 n_heads=2,
                 dropout=0.1):
        super(Conformer, self).__init__()

        self.layers = nn.ModuleList([
            ConformerLayer(
                channels=channels,
                n_heads=n_heads,
                dropout=dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, x, pos_emb, x_mask):
        for layer in self.layers:
            x = layer(x, pos_emb, x_mask)
        return x
