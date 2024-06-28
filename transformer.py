from dataclasses import dataclass
from data import Batch
import math
import os
import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")


# Defaults match the config for bitstrings
@dataclass
class Config:
    vocab_size: int
    window_size: int
    num_layers: int = 6
    num_heads: int = 6
    embed_dim: int = 384

    def scale_init(self):
        "Shrinks init for c_proj layers in the model."
        return math.sqrt(2 * self.num_layers)


class MLP(nn.Module):
    """
    Multi-layer perceptron. Has the same size for both input and output.
    No dropout but maybe we could stick some at the end.
    """

    def __init__(self, config, dropout=None):
        super().__init__()
        assert dropout is not None
        self.c_fc = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False)
        nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=False)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / config.scale_init())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input is (batch_size, embed_dim, ...any).
        Output is (batch_size, embed_dim, ...any).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    """
    Just the self-attention mechanism.
    """

    def __init__(self, config, causal=None, dropout=None):
        super().__init__()
        assert causal is not None
        assert dropout is not None
        self.config = config
        self.causal = causal
        self.dropout = dropout
        assert config.embed_dim % config.num_heads == 0
        self.dim_per_head = config.embed_dim // config.num_heads
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        nn.init.normal_(self.c_attn.weight, mean=0.0, std=0.02)
        # output projection
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / config.scale_init())
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input is (batch_size, num_tokens, embed_dim).
        Output is (batch_size, num_tokens, embed_dim).

        num_tokens can be anything <= window_size.
        """
        batch_size, num_tokens, embed_dim = x.size()
        assert num_tokens <= self.config.window_size
        assert embed_dim == self.config.embed_dim

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(embed_dim, dim=2)
        k = k.view(
            batch_size,
            num_tokens,
            self.config.num_heads,
            self.dim_per_head,
        ).transpose(1, 2)
        q = q.view(
            batch_size,
            num_tokens,
            self.config.num_heads,
            self.dim_per_head,
        ).transpose(1, 2)
        v = v.view(
            batch_size,
            num_tokens,
            self.config.num_heads,
            self.dim_per_head,
        ).transpose(1, 2)

        # q, k, v are now all:
        # (batch_size, num_heads, window_size, dim_per_head)

        # Use the Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=self.causal,
        )

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch_size, num_tokens, embed_dim)

        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class SelfAttentionLayer(nn.Module):
    """
    A single layer of a self-attention transformer.
    Takes a flag of whether to causal mask.
    """

    def __init__(self, config, causal=None, dropout=None):
        super().__init__()
        assert causal is not None
        assert dropout is not None
        self.norm1 = nn.LayerNorm(config.embed_dim, bias=False)
        self.self_attn = SelfAttention(config, causal=causal, dropout=dropout)
        self.norm2 = nn.LayerNorm(config.embed_dim, bias=False)
        self.mlp = MLP(config, dropout=dropout)

    def forward(self, x):
        """
        Input is (batch_size, window_size, embed_dim).
        Output is (batch_size, window_size, embed_dim).
        """
        x = x + self.self_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SelfAttentionStack(nn.Module):
    """
    A stack of self-attention layers.
    Takes a flag of whether to causal mask.
    """

    def __init__(self, config, causal=None, dropout=None):
        super().__init__()
        assert causal is not None
        assert dropout is not None
        self.layers = nn.ModuleList(
            [
                SelfAttentionLayer(config, causal=causal, dropout=dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim, bias=False)

    def forward(self, x):
        """
        Input is (batch_size, window_size, embed_dim).
        Output is (batch_size, window_size, embed_dim).
        """
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class DecoderOnly(nn.Module):
    """
    Decoder-only transformer, including embedding layers.
    """

    def __init__(self, config: Config, dropout=None):
        super().__init__()
        assert dropout is not None
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.position_embedding = nn.Embedding(config.window_size, config.embed_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.stack = SelfAttentionStack(config, causal=True, dropout=0.0)

        # Weight tying
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, batch):
        """
        batch is a Batch that may or may not have outputs.

        Returns (logits, loss).

        logits is (batch size, window size, vocab size).

        If outputs are provided in the batch, we use it to calculate a scalar loss.
        If outputs are not provided, loss is just None.
        """
        _, num_tokens = batch.inputs.size()
        if num_tokens != self.config.window_size:
            raise ValueError(
                f"Window size is {self.config.window_size}, got {num_tokens} tokens"
            )
        positions = torch.arange(
            0, num_tokens, dtype=torch.long, device=batch.inputs.device
        )
        embedded_tokens = self.token_embedding(batch.inputs)
        embedded_positions = self.position_embedding(positions)
        x = embedded_tokens + embedded_positions
        x = self.stack(x)

        logits = self.head(x)
        loss = None

        if batch.outputs is not None:
            # Calculate the loss
            flat_loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch.outputs.view(-1),
                ignore_index=-1,
                reduction="none",
            )
            unweighted_loss = flat_loss.view(batch.outputs.shape)

            weights = torch.where(
                batch.priorities, torch.tensor(0.8), torch.tensor(0.2)
            )
            weighted_loss = unweighted_loss * weights
            loss = weighted_loss.mean()

        return logits, loss

    @torch.no_grad()
    def generate(self, input_tokens, max_new_tokens=20, temperature=1.0, encoding=None):
        """
        tokens is a list of tokens.
        If encoding is provided, use it to display debug information.
        """
        tokens = list(input_tokens)
        assert not self.training
        for _ in range(max_new_tokens):
            usable = tokens[-self.config.window_size :]
            while len(usable) < self.config.window_size:
                usable = [0] + usable
            monobatch = Batch.monobatch(usable)
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(monobatch)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            if encoding is not None:
                # Debug
                print("probabilities:")
                for token, prob in enumerate(probs[0]):
                    s = encoding.decode([token])
                    print(f"{s}: {prob:.3f}")
                print("chosen token:", encoding.decode(next_token[0].tolist()))

            next_token = next_token.item()
            tokens.append(next_token)
            if next_token == 0:
                break

        return tokens


def make_optimizer(module, weight_decay, learning_rate, betas, device_type):
    "Completely from nanoGPT."
    # start with all of the candidate parameters
    param_dict = {name: param for name, param in module.named_parameters()}
    # filter out those that do not require grad
    param_dict = {
        name: param for name, param in param_dict.items() if param.requires_grad
    }
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = []
    nodecay_params = []
    for name, param in sorted(param_dict.items()):
        if param.dim() < 2:
            nodecay_params.append(param)
        else:
            decay_params.append(param)
        # print(f"{name}: {tuple(param.shape)}")
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, "
        f"with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, "
        f"with {num_nodecay_params:,} parameters"
    )
    # Create fused AdamW optimizer
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, fused=True
    )

    return optimizer
