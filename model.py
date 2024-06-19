import torch
import torch.nn as nn

# Matching the config for bitstrings
NUM_LAYERS = 6
NUM_HEADS = 6
NUM_EMBED = 384


class MLP(nn.Module):
    """
    Multi-layer perceptron. Has the same size for both input and output.
    No dropout but maybe we could stick some at the end.
    """
    def __init__(self, size):
        super().__init__()
        self.expansion    = nn.Linear(size, 4 * size)
        self.gelu    = nn.GELU()
        self.projection  = nn.Linear(4 * size, size)

    def forward(self, x):
        x = self.expansion(x)
        x = self.gelu(x)
        x = self.projection(x)
        return x


class Block(nn.module):
    """
    A transformer block.
    """
    def __init__(self, num_embed, num_heads):
        raise "TODO"


class DecoderOnly(nn.Module):
    """
    Decoder only transformer, with embedding layers.
    """
    def __init__(self, num_features, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([MLP(num_features) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x