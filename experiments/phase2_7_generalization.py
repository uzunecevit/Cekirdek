#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 2.7: Generalization + Stabilite + Ablation

Testler:
1. Generalization: train loss ↓ ama val loss ↓ mü?
2. Stabilite: 100→500→1000 update sonrası diverge var mı?
3. Ablation: STDP vs Backprop vs İkisi birlikte
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SpikingLM

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_TRAIN = 30
N_VAL = 20
N_UPDATES = [100, 500, 1000]
LR = 0.002


def load_model():
    path = os.path.join(CKPT_DIR, "spiking_lm.pt")
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
        decay=0.3,
        use_surrogate=False,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_splits(ckpt):
    import random

    names_path = os.path.join(
        os.path.dirname(__file__), "../../VİCDAN_HEBBIAN/micro/names.txt"
    )
    with open(names_path) as f:
        names = [line.strip() for line in f if line.strip()]
    random.seed(42)
    random.shuffle(names)
    stoi = ckpt["stoi"]
    BOS = ckpt["BOS"]

    train_names = names[:100]
    val_names = names[100:200]

    def make_seqs(name_list):
        seqs = []
        for name in name_list:
            tokens = [BOS] + [stoi[c] for c in name] + [BOS]
            if len(tokens) >= 2:
                seqs.append((tokens[:-1], tokens[1:]))
        return seqs

    return make_seqs(train_names)[:N_TRAIN], make_seqs(val_names)[:N_VAL]


def compute_loss(model, sequences):
    losses = []
    with torch.no_grad():
        for x, y in sequences:
            idx = torch.tensor([x], device=DEVICE)
            target = torch.tensor([y], device=DEVICE)
            _, loss = model(idx, targets=target)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def apply_stdp(model, x, y, lr):
    with torch.no_grad():
        idx = torch.tensor([x], device=DEVICE)
        target = torch.tensor([y], device=DEVICE)
        logits, loss = model(idx, targets=target)
        vocab_size = logits.size(-1)

        for t in range(logits.size(1)):
            probs = F.softmax(logits[0, t, :], dim=-1)
            target_token = y[t]
            target_onehot = torch.zeros(vocab_size, device=DEVICE)
            target_onehot[target_token] = 1.0
            token_idx = x[min(t, len(x) - 1)]
            pre = model.wte.weight[token_idx]
            error = target_onehot - probs
            dw = lr * error.unsqueeze(1) * pre.unsqueeze(0)
            model.lm_head.weight.add_(dw)
            model.lm_head.weight.clamp_(-1.0, 1.0)
        return loss.item()


def apply_backprop(model, x, y, lr, optimizer):
    idx = torch.tensor([x], device=DEVICE)
    target = torch.tensor([y], device=DEVICE)
    _, loss = model(idx, targets=target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def test_generalization():
    """Test 1: Train loss ↓ ama val loss ↓ mü?"""
    print("\n" + "=" * 60)
    print("TEST 1: Generalization (Train vs Val Loss)")
    print("=" * 60)

    train_seqs, val_seqs = load_splits(load_model()[1])
    model, ckpt = load_model()

    initial_train = compute_loss(model, train_seqs)
    initial_val = compute_loss(model, val_seqs)
    print(f"  Initial — Train: {initial_train:.4f}, Val: {initial_val:.4f}")

    # STDP ile güncelle (sadece train)
    for step in range(200):
        x, y = train_seqs[step % len(train_seqs)]
        apply_stdp(model, x, y, LR)

    final_train = compute_loss(model, train_seqs)
    final_val = compute_loss(model, val_seqs)

    train_change = (final_train - initial_train) / initial_train * 100
    val_change = (final_val - initial_val) / initial_val * 100

    print(f"  Final  — Train: {final_train:.4f} ({train_change:+.1f}%)")
    print(f"  Final  — Val:   {final_val:.4f} ({val_change:+.1f}%)")

    if val_change < 0:
        print(f"\n  ✅ Gerçek öğrenme: val loss da düştü")
    elif val_change < 1:
        print(f"\n  ⚠️ Zayıf sinyal: val loss neredeyse aynı")
    else:
        print(f"\n  ❌ Overfitting: val loss arttı")


def test_stability():
    """Test 2: 100→500→1000 update sonrası stabil mi?"""
    print("\n" + "=" * 60)
    print("TEST 2: Stabilite (Uzun Vadeli Öğrenme)")
    print("=" * 60)

    train_seqs, val_seqs = load_splits(load_model()[1])

    for n_updates in N_UPDATES:
        model, ckpt = load_model()
        initial_train = compute_loss(model, train_seqs)
        initial_val = compute_loss(model, val_seqs)

        for step in range(n_updates):
            x, y = train_seqs[step % len(train_seqs)]
            apply_stdp(model, x, y, LR)

        final_train = compute_loss(model, train_seqs)
        final_val = compute_loss(model, val_seqs)

        print(
            f"  Updates={n_updates:4d}: train={initial_train:.4f}→{final_train:.4f}, val={initial_val:.4f}→{final_val:.4f}"
        )


def test_ablation():
    """Test 3: STDP vs Backprop vs İkisi birlikte"""
    print("\n" + "=" * 60)
    print("TEST 3: Ablation (STDP vs Backprop vs Hybrid)")
    print("=" * 60)

    train_seqs, val_seqs = load_splits(load_model()[1])
    N = 200

    # --- STDP only ---
    model, ckpt = load_model()
    for step in range(N):
        x, y = train_seqs[step % len(train_seqs)]
        apply_stdp(model, x, y, LR)
    stdp_train = compute_loss(model, train_seqs)
    stdp_val = compute_loss(model, val_seqs)

    # --- Backprop only ---
    model, ckpt = load_model()
    optimizer = torch.optim.Adam(model.lm_head.parameters(), lr=LR)
    for step in range(N):
        x, y = train_seqs[step % len(train_seqs)]
        apply_backprop(model, x, y, LR, optimizer)
    bp_train = compute_loss(model, train_seqs)
    bp_val = compute_loss(model, val_seqs)

    # --- Hybrid (STDP + Backprop) ---
    model, ckpt = load_model()
    optimizer = torch.optim.Adam(model.lm_head.parameters(), lr=LR * 0.1)
    for step in range(N):
        x, y = train_seqs[step % len(train_seqs)]
        apply_stdp(model, x, y, LR)
        apply_backprop(model, x, y, LR * 0.1, optimizer)
    hybrid_train = compute_loss(model, train_seqs)
    hybrid_val = compute_loss(model, val_seqs)

    initial_train = compute_loss(load_model()[0], train_seqs)
    initial_val = compute_loss(load_model()[0], val_seqs)

    print(f"  Initial:      train={initial_train:.4f}, val={initial_val:.4f}")
    print(f"  STDP only:    train={stdp_train:.4f}, val={stdp_val:.4f}")
    print(f"  Backprop only: train={bp_train:.4f}, val={bp_val:.4f}")
    print(f"  Hybrid:       train={hybrid_train:.4f}, val={hybrid_val:.4f}")

    # Karşılaştırma
    methods = [
        ("STDP only", stdp_val),
        ("Backprop only", bp_val),
        ("Hybrid", hybrid_val),
    ]
    best = min(methods, key=lambda x: x[1])
    print(f"\n  En iyi val loss: {best[0]} ({best[1]:.4f})")


def main():
    print("=" * 60)
    print("VİCDAN_SPIKE — Phase 2.7: Generalization + Stabilite + Ablation")
    print("=" * 60)

    test_generalization()
    test_stability()
    test_ablation()

    print("\n" + "=" * 60)
    print("TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()
