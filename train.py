import data
import model
import numpy as np
import torch


eval_interval = 250
eval_iters = 200
log_interval = 10
batch_size = 128
learning_rate = 1e-3  # with baby networks can afford to go a bit higher

max_iters = 100000
lr_decay_iters = max_iters  # make equal to max_iters usually

min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

encoding = data.generate()
config = model.Config(vocab_size=encoding.vocab_size)
tokens_per_iter = batch_size * config.window_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
torch.manual_seed(1337)
assert torch.cuda.is_bf16_supported()
dtype = torch.bfloat16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ptdtype = torch.bfloat16
context = torch.amp.autocast(device_type="cuda", dtype=dtype)


# Essentially a data loader
def get_batch(tokens):
    start_indices = torch.randint(0, len(tokens) - config.window_size, (batch_size,))
    inputs = torch.stack(
        [
            torch.from_numpy((data[i : i + config.window_size]).astype(np.int64))
            for i in start_indices
        ]
    )
    outputs = torch.stack(
        [
            torch.from_numpy(
                (data[i + 1 : i + config.window_size + 1]).astype(np.int64)
            )
            for i in start_indices
        ]
    )
    inputs = inputs.pin_memory().to("cuda", non_blocking=True)
    outputs = outputs.pin_memory().to("cuda", non_blocking=True)
    return inputs, outputs


iterations = 0
best_val_loss = 1e9
print("initializing a new model from scratch...")
model = model.DecoderOnly(config)
model.to("cuda")
