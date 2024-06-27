from dataclasses import dataclass
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

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False)
        nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=False)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / config.scale_init())

    def forward(self, x):
        """
        Input is (batch_size, embed_dim, ...any).
        Output is (batch_size, embed_dim, ...any).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class SelfAttention(nn.Module):
    """
    Just the self-attention mechanism.
    """

    def __init__(self, config, causal):
        super().__init__()
        self.config = config
        self.causal = causal
        assert config.embed_dim % config.num_heads == 0
        self.dim_per_head = config.embed_dim // config.num_heads
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        nn.init.normal_(self.c_attn.weight, mean=0.0, std=0.02)
        # output projection
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02 / config.scale_init())

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
            dropout_p=False,
            is_causal=self.causal,
        )

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch_size, num_tokens, embed_dim)

        # output projection
        y = self.c_proj(y)
        return y


class SelfAttentionLayer(nn.Module):
    """
    A single layer of a self-attention transformer.
    Takes a flag of whether to causal mask.
    """

    def __init__(self, config, causal):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, bias=False)
        self.self_attn = SelfAttention(config, causal)
        self.norm2 = nn.LayerNorm(config.embed_dim, bias=False)
        self.mlp = MLP(config)

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

    def __init__(self, config, causal):
        super().__init__()
        self.layers = nn.ModuleList(
            [SelfAttentionLayer(config, causal) for _ in range(config.num_layers)]
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


class CrossAttentionLayer(nn.Module):
    """
    A layer of the decoder of an encoder-decoder transformer.
    Combines self attention and cross attention.
    """

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, bias=False)
        self.self_attn = SelfAttention(config, True)
        self.norm2 = nn.LayerNorm(config.embed_dim, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            config.embed_dim, config.num_heads, bias=False, batch_first=True
        )
        self.norm3 = nn.LayerNorm(config.embed_dim, bias=False)
        self.mlp = MLP(config)

    def forward(self, encoded, x):
        """
        encoded is (batch_size, window_size, embed_dim).
        x is (batch_size, window_size, embed_dim).
        output is (batch_size, window_size, embed_dim).
        """
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), encoded, encoded, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class CrossAttentionStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.embed_dim, bias=False)

    def forward(self, encoded, x):
        for layer in self.layers:
            x = layer(encoded, x)
        x = self.norm(x)
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
        self.stack = SelfAttentionStack(config, causal=True)

        # Weight tying
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens, targets=None, priority=None):
        """
        Input is (batch_size, window_size).
        Output is (batch_size, num_outputs, vocab_size)

        If targets are provided, we provide an output for each target, and a scalar loss.
        Targets is (batch_size, num_outputs).
        Otherwise, there's just one output, and loss is None.

        If priority is provided, it's a boolean mask of the targets for which ones are important.
        """
        _, num_tokens = tokens.size()
        if num_tokens > self.config.window_size:
            raise ValueError(
                f"Window size is {self.config.window_size}, got {num_tokens} tokens"
            )
        positions = torch.arange(0, num_tokens, dtype=torch.long, device=tokens.device)
        embedded_tokens = self.token_embedding(tokens)
        embedded_positions = self.position_embedding(positions)
        x = embedded_tokens + embedded_positions
        x = self.stack(x)

        if targets is not None:
            # Also calculate the loss
            logits = self.head(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction="none",
            )
            loss = loss.view(targets.shape)
            # TODO: try prioritizing here
            loss = loss.mean()
        else:
            # Only forward the head on the very last position
            logits = self.head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=20, temperature=1.0, encoding=None):
        """
        If encoding is provided, use it to display debug information.
        """
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
            if encoding is not None:
                # Debug
                print("probabilities:")
                for token, prob in enumerate(probs[0]):
                    s = encoding.decode([token])
                    print(f"{s}: {prob:.3f}")
                print("chosen token:", encoding.decode(next_token[0].tolist()))
            # append sampled index to the running sequence and continue
            tokens = torch.cat((tokens, next_token), dim=1)
            if next_token == 0:
                break

        return tokens


class EncoderDecoder(nn.Module):
    """
    Encoder-decoder transformer, including embedding layers.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        self.in_pos_embed = nn.Embedding(config.window_size, config.embed_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.out_pos_embed = nn.Embedding(config.window_size, config.embed_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.encoder = SelfAttentionStack(config, causal=False)
        self.decoder = CrossAttentionStack(config)

        # Weight tying
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_tokens, output_tokens, targets=None):
        """
        input_tokens is (batch_size, window_size).
        output_tokens is (batch_size, num_outputs).
        final output is (batch_size, num_outputs, vocab_size)

        If targets are provided, we provide a final output for each target, and a scalar loss.
        Otherwise, there's just one output, and loss is None.
        """
        _, num_input_tokens = input_tokens.size()
        assert num_input_tokens == self.config.window_size
        input_pos = torch.arange(
            0, num_input_tokens, dtype=torch.long, device=input_tokens.device
        )
        _, num_output_tokens = output_tokens.size()
        assert num_output_tokens <= self.config.window_size
        output_pos = torch.arange(
            0, num_output_tokens, dtype=torch.long, device=output_tokens.device
        )
        input_embed = self.token_embed(input_tokens) + self.in_pos_embed(input_pos)
        encoded = self.encoder(input_embed)
        output_embed = self.token_embed(output_tokens) + self.out_pos_embed(output_pos)
        final_layer = self.decoder(encoded, output_embed)

        if targets is not None:
            # Also calculate the loss
            logits = self.head(final_layer)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Only forward the head on the very last position
            logits = self.head(final_layer[:, -1, :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, input_tokens, output_tokens, max_new_tokens=20, temperature=1.0):
        assert not self.training
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long, bail
            if output_tokens.size(1) > self.config.window_size:
                break
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(input_tokens, output_tokens)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            output_tokens = torch.cat((output_tokens, next_token), dim=1)
            if next_token == 0:
                break
        return output_tokens


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
