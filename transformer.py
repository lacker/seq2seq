from dataclasses import dataclass
import inspect
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
    num_layers: int = 6
    num_heads: int = 6
    embed_dim: int = 384
    window_size: int = 16

    def scale_init(self):
        "Shrinks init for c_proj layers in the model."
        return math.sqrt(2 * self.num_layers)


class MLP(nn.Module):
    """
    Multi-layer perceptron. Has the same size for both input and output.
    No dropout but maybe we could stick some at the end.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False)
        nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=False)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / config.scale_init())

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, config, causal=None):
        super().__init__()
        # causal must be set explicitly
        assert causal is not None
        self.causal = causal
        assert config.embed_dim % config.num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        nn.init.normal_(self.c_attn.weight, mean=0.0, std=0.02)
        # output projection
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / config.scale_init())
        # regularization
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (embed_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Use the Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=False,
            is_causal=self.causal,
        )

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, bias=False)
        self.csa = SelfAttention(config, causal=True)
        self.norm2 = nn.LayerNorm(config.embed_dim, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.csa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderOnly(nn.Module):
    """
    Decoder-only transformer, including embedding layers.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.position_embedding = nn.Embedding(config.window_size, config.embed_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.embed_dim, bias=False)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

        # Weight tying
        self.token_embedding.weight = self.head.weight

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
        embedded_tokens = self.token_embedding(tokens)
        embedded_positions = self.position_embedding(positions)
        x = embedded_tokens + embedded_positions
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if targets is not None:
            # Also calculate the loss
            logits = self.head(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Only forward the head on the very last position
            logits = self.head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0):
        assert not self.training
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            cropped = (
                tokens
                if tokens.size(1) <= self.config.window_size
                else tokens[:, -self.config.window_size :]
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        "Completely from nanoGPT."
        # start with all of the candidate parameters
        param_dict = {name: param for name, param in self.named_parameters()}
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
            print(f"{name}: {tuple(param.shape)}")
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
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer
