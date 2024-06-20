import torch
import torch.nn as nn

# Matching the config for bitstrings
NUM_LAYERS = 6
NUM_HEADS = 6
EMBED_DIM = 384


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


class CausalSelfAttention(nn.Module):
    """
    Just MultiheadAttention, adding a causal mask.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        seq_len, _batch_size, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        # Create the causal mask (upper triangular with ones above the diagonal)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        output, _ = self.attention(x, x, x, attn_mask=mask)
        return output


class Block(nn.module):
    """
    A single transformer block.
    """
    def __init__(self, embed_dim, num_heads):
        self.norm1 = nn.LayerNorm(embed_dim)
        self.csa = CausalSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        x = x + self.csa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderOnly(nn.Module):
    """
    Decoder-only transformer, including embedding layers.
    """
    def __init__(self, num_features, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([MLP(num_features) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x