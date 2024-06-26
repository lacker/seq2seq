import data
import random
import torch
import transformer

checkpoint = torch.load(transformer.checkpoint_path, map_location="cuda")
config = checkpoint["config"]
model = transformer.DecoderOnly(config, dropout=0.0)
state_dict = checkpoint["model"]

# I think this is because we're saving a model using DDP and we want to load it without DDP.
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to("cuda")

context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
encoding = data.Dataset(config.window_size)

score = 0
num_samples = 100
random.seed(1234569)
with torch.no_grad():
    with context:
        for i in range(num_samples):
            target = data.generate_one()
            question, right_answer = target.split("=")
            print("question:", question)
            print("right answer:", right_answer)
            start = question + "="
            while len(start) < config.window_size:
                start = "." + start
            input_tokens = encoding.encode(start)
            output_tokens = model.generate(
                input_tokens, temperature=0.001, encoding=None
            )
            output_str = encoding.decode(output_tokens)
            model_answer = f"malformed answer: {repr(output_str)}"
            if "=" in output_str:
                post_eq = output_str.split("=", 1)[1]
                if "." in post_eq:
                    model_answer = post_eq.split(".", 1)[0]
            print("model answer:", model_answer)
            if model_answer == right_answer:
                score += 1
                print("correct")
            else:
                print("incorrect")
            print("-----------------")

print(f"final score: {score}/{num_samples}")
