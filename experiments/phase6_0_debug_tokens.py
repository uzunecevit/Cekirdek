#!/usr/bin/env python3
"""Debug: token indices'i kontrol et."""

import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from src.model import SpikingLM

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(os.path.join(DATA_DIR, "vocab.json")) as f:
    vocab = json.load(f)
char2idx = vocab["char2idx"]
idx2char = {str(v): k for k, v in char2idx.items()}

text = "3+4="
target_text = "3+4=7"

x = [char2idx[c] for c in text if c in char2idx]
y = [char2idx[c] for c in target_text if c in char2idx]

print(f"Input: {text!r}")
print(f"  x = {x}")
print(f"  x chars = {[idx2char[str(i)] for i in x]}")
print(f"\nTarget: {target_text!r}")
print(f"  y = {y}")
print(f"  y chars = {[idx2char[str(i)] for i in y]}")
print(f"\n  y[-1] = {y[-1]} → char = {idx2char[str(y[-1])]}")
print(f"  len(x) = {len(x)}, len(y) = {len(y)}")

# Model forward
path = os.path.join(CKPT_DIR, "spiking_lm_v2.pt")
ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
config = ckpt["config"]
model = SpikingLM(
    vocab_size=ckpt["vocab_size"],
    d_model=config["d_model"],
    n_layer=config["n_layer"],
    n_head=config["n_head"],
    d_ff=config["d_ff"],
    block_size=config["block_size"],
    threshold=0.3,
    decay=0.1,
    amplitude=2.0,
    use_surrogate=True,
    ternary=True,
    fast_weight_fc2=True,
).to(DEVICE)
model.load_state_dict(ckpt["model_state"])

idx = torch.tensor([x], device=DEVICE)
logits, loss = model(idx)

print(f"\nLogits shape: {logits.shape}")
print(f"  T = {logits.size(1)}")

for t in range(logits.size(1)):
    probs = F.softmax(logits[0, t], dim=-1)
    top3 = probs.topk(3)
    print(f"\n  t={t}: input char = {idx2char[str(x[t])]}")
    for i in range(3):
        idx_i = top3.indices[i].item()
        prob_i = top3.values[i].item()
        print(f"    {idx2char[str(idx_i)]!r}: {prob_i:.4f}")
