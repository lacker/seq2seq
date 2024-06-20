import model

config = model.Config()

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

tokens_per_iter = batch_size * config.window_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
