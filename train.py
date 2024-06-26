import data
import math
import numpy as np
import time
import torch
import transformer


def get_batch(inputs, outputs, batch_size):
    """
    Inputs is: (batch_size, window_size)
    Outputs is: (batch_size, window_size)
    Returns (inputs, outputs).
    """
    assert len(inputs) == len(outputs)
    indices = np.random.randint(0, len(inputs), (batch_size,))
    inputs = torch.from_numpy(inputs[indices].astype(np.int64))
    outputs = torch.from_numpy(outputs[indices].astype(np.int64))
    inputs = inputs.pin_memory().to("cuda", non_blocking=True)
    outputs = outputs.pin_memory().to("cuda", non_blocking=True)
    return inputs, outputs


@torch.no_grad()
def estimate_loss(train_inputs, train_outputs, val_inputs, val_outputs):
    "Returns a (train_loss, val_loss) tuple."
    answer = [None, None]
    model.eval()
    for i, (all_inputs, all_outputs) in enumerate(
        [(train_inputs, train_outputs), (val_inputs, val_outputs)]
    ):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            inputs, outputs = get_batch(all_inputs, all_outputs, batch_size)
            with context:
                logits, loss = model(inputs, outputs)
            losses[k] = loss.item()
        answer[i] = losses.mean()
    model.train()
    return tuple(answer)


def get_learning_rate(i, lr_decay_iters, min_lr):
    "Get the learning rate to use for the ith iteration."
    warmup_iters = 100  # not super necessary potentially

    if i < warmup_iters:
        return base_learning_rate * i / warmup_iters
    elif i > lr_decay_iters:
        return min_lr

    # use "cosine decay"
    decay_ratio = (i - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    assert 0 <= coeff <= 1
    return min_lr + (base_learning_rate - min_lr) * coeff


if __name__ == "__main__":
    eval_interval = 250
    eval_iters = 200
    log_interval = 10
    batch_size = 256
    base_learning_rate = 1e-3  # with baby networks can afford to go a bit higher

    max_iters = 30000
    lr_decay_iters = max_iters  # make equal to max_iters usually
    weight_decay = 1e-1
    min_lr = 1e-4  # learning_rate / 10 usually
    beta1 = 0.9
    beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
    grad_clip = 1.0

    encoding = data.generate()
    config = transformer.Config(vocab_size=encoding.vocab_size)
    (train_inputs, train_outputs), (val_inputs, val_outputs) = encoding.make_data(
        config.window_size
    )
    tokens_per_iter = batch_size * config.window_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    torch.manual_seed(1337)
    assert torch.cuda.is_bf16_supported()
    context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    iterations = 0
    best_val_loss = 1e9
    print("initializing a new model from scratch...")
    model = transformer.DecoderOnly(config)
    model.to("cuda")

    optimizer = model.configure_optimizers(
        weight_decay, base_learning_rate, (beta1, beta2), "cuda"
    )

    print("compiling the model...")
    model = torch.compile(model)

    # The training loop.
    # Start by fetching the very first batch.
    x, y = get_batch(train_inputs, train_outputs, batch_size)
    current_time = time.time()
    for step in range(max_iters):
        # Determine the learning rate for this iteration
        lr = get_learning_rate(step, lr_decay_iters, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Evaluate the loss
        if step > 0 and step % eval_interval == 0:
            train_loss, val_loss = estimate_loss(
                train_inputs, train_outputs, val_inputs, val_outputs
            )
            print(f"{step=} {train_loss=:.3f} {val_loss=:.3f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("saving checkpoint...")
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                torch.save(checkpoint, transformer.checkpoint_path)

        # Forward pass
        with context:
            logits, loss = model(x, y)
        # Should async prefetch
        x, y = get_batch(train_inputs, train_outputs, batch_size)
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        # Flush gradients to free up memory
        optimizer.zero_grad(set_to_none=True)

        # Timing and logging
        next_time = time.time()
        batch_time = next_time - current_time
        current_time = next_time
        if step % log_interval == 0:
            # Note: this is a CPU-GPU sync point.
            batch_loss = loss.item()
            print(f"{step=} {batch_loss=:.3f}, batch time {batch_time*1e3:.1f}ms")
