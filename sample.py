import torch
import transformer

checkpoint = torch.load(transformer.checkpoint_path, map_location="cuda")
model = transformer.DecoderOnly(checkpoint["config"])
