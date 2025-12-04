import torch

ckpt_path = "experiments/tiny_transformer/run3/checkpoint.pt"  # <- your file name
ckpt = torch.load(ckpt_path, map_location="cpu")

print(type(ckpt))
if isinstance(ckpt, dict):
    print("Keys:", ckpt.keys())

