from dataclasses import dataclass
import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron. Has the same size for both input and output.
    No dropout but maybe we could stick some at the end.
    """

    def __init__(self, size):
        super().__init__()
        self.expansion = nn.Linear(size, 4 * size)
        self.gelu = nn.GELU()
        self.projection = nn.Linear(4 * size, size)

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


# Defaults match the config for bitstrings
@dataclass
class Config:
    num_layers: int = 6
    num_heads: int = 6
    embed_dim: int = 384
    window_size: int = 16
    vocab_size: int = 16


def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class DecoderOnly(nn.Module):
    """
    Decoder-only transformer, including embedding layers.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.modules = nn.ModuleDict(
            dict(
                token_embedding=nn.Embedding(config.vocab_size, config.embed_dim),
                position_embedding=nn.Embedding(config.window_size, config.embed_dim),
                blocks=nn.ModuleList(
                    [
                        Block(config.embed_dim, config.num_heads)
                        for _ in range(config.num_layers)
                    ]
                ),
                norm=nn.LayerNorm(config.embed_dim),
                head=nn.Linear(config.embed_dim, config.vocab_size, bias=False),
            )
        )

        # Weight tying
        self.modules.token_embedding.weight = self.modules.head.weight

        self.apply(init_weights)

        # "special scaled init" as per nanogpt / gpt2.
        for pn, p in self.named_parameters():
            if pn.endswith("projection.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers)
                )

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens, targets=None):
        "If targets are provided, also calculate the loss."
        _, num_tokens = tokens.size()
        if num_tokens > self.config.window_size:
            raise ValueError(
                f"Window size is {self.config.window_size}, got {num_tokens} tokens"
            )
        positions = torch.arange(0, num_tokens, dtype=torch.long, device=tokens.device)
        embedded_tokens = self.modules.token_embedding(tokens)
        embedded_positions = self.modules.position_embedding(positions)
        x = embedded_tokens + embedded_positions
        for block in self.modules.blocks:
            x = block(x)
        x = self.modules.norm(x)

        if targets is not None:
            # Also calculate the loss
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0):
        assert not self.training
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            cropped = (
                tokens
                if tokens.size(1) <= self.config.block_size
                else tokens[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(cropped)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens
