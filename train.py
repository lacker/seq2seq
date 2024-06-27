import data
import math
import time
import torch
import transformer


@torch.no_grad()
def estimate_loss(model, context, eval_iters, batch_size, train, val):
    "Returns a (train_loss, val_loss) tuple."
    answer = [None, None]
    model.eval()
    for i, subset in enumerate([train, val]):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch = subset.get_batch(batch_size)
            with context:
                logits, loss = model(batch)
            losses[k] = loss.item()
        answer[i] = losses.mean()
    model.train()
    return tuple(answer)


def get_learning_rate(step, base_learning_rate, lr_decay_iters, min_lr):
    "Get the learning rate to use for the ith iteration."
    warmup_iters = 100  # not super necessary potentially

    if step < warmup_iters:
        return base_learning_rate * step / warmup_iters
    elif step > lr_decay_iters:
        return min_lr

    # use "cosine decay"
    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    assert 0 <= coeff <= 1
    return min_lr + (base_learning_rate - min_lr) * coeff


def main():
    eval_interval = 250
    eval_iters = 200
    log_interval = 10
    batch_size = 256
    base_learning_rate = 1e-3  # with baby networks can afford to go a bit higher

    max_iters = 50000
    lr_decay_iters = max_iters  # make equal to max_iters usually
    weight_decay = 1e-1
    min_lr = 1e-4  # learning_rate / 10 usually
    beta1 = 0.9
    beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
    grad_clip = 1.0
    window_size = 16

    dataset = data.generate_dataset(window_size)

    config = transformer.Config(
        vocab_size=dataset.vocab_size, window_size=dataset.window_size
    )
    tokens_per_iter = batch_size * config.window_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    torch.manual_seed(1337)
    assert torch.cuda.is_bf16_supported()
    context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    best_val_loss = 1e9
    print("initializing a new model from scratch...")
    model = transformer.DecoderOnly(config, dropout=0.1)
    model.to("cuda")

    optimizer = transformer.make_optimizer(
        model, weight_decay, base_learning_rate, (beta1, beta2), "cuda"
    )

    print("compiling the model...")
    model = torch.compile(model)

    # The training loop.
    # Start by fetching the very first batch.
    batch = dataset.train.get_batch(batch_size)
    current_time = time.time()
    for step in range(max_iters):
        # Determine the learning rate for this iteration
        lr = get_learning_rate(step, base_learning_rate, lr_decay_iters, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Evaluate the loss
        if step > 0 and step % eval_interval == 0:
            train_loss, val_loss = estimate_loss(
                model,
                context,
                eval_iters,
                batch_size,
                dataset.train,
                dataset.val,
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
            logits, loss = model(batch)
        # Should async prefetch
        batch = dataset.train.get_batch(batch_size)
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


if __name__ == "__main__":
    main()
