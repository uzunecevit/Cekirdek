#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 2.6: STDP Ölçeklenebilirlik Testi

Amaç: "Dense R-STDP sinyali ölçeklenebilir mi?"

Testler:
1. STDP vs No-STDP (aynı batch, karşılaştırma)
2. Layer karşılaştırma: FFN vs LM Head vs ikisi birlikte
3. Learning rate scaling: η = 0.0005 → 0.005 → 0.02
4. Multi-step: aynı batch → 1/5/10 tekrar STDP
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import SpikingLM

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test config
N_STEPS = 50
N_SAMPLES = 50
LR_VALUES = [0.0005, 0.002, 0.01]
REPEAT_VALUES = [1, 5, 10]


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


def load_dataset(ckpt, n_samples=N_SAMPLES):
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
    sequences = []
    for name in names[:n_samples]:
        tokens = [BOS] + [stoi[c] for c in name] + [BOS]
        if len(tokens) >= 2:
            sequences.append((tokens[:-1], tokens[1:]))
    return sequences[:n_samples]


def compute_loss(model, sequences):
    losses = []
    with torch.no_grad():
        for x, y in sequences:
            idx = torch.tensor([x], device=DEVICE)
            target = torch.tensor([y], device=DEVICE)
            _, loss = model(idx, targets=target)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def apply_stdp(model, x, y, lr, update_layer="lm_head"):
    """STDP update uygula."""
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

            # Pre: input embedding
            token_idx = x[min(t, len(x) - 1)]
            pre = model.wte.weight[token_idx]

            # Gradient-like update
            error = target_onehot - probs
            dw = lr * error.unsqueeze(1) * pre.unsqueeze(0)

            if update_layer == "lm_head":
                model.lm_head.weight.add_(dw)
                model.lm_head.weight.clamp_(-1.0, 1.0)
            elif update_layer == "ffn":
                # FFN fc2 weight
                w = model.blocks[0].ffn.fc2.linear.weight
                # dw shape'i uyarla: (d_model, d_ff) → hata projeksiyonu
                # Basit: LM head dw'yi FFN'e uygula (yaklaşık)
                w.add_(dw[:, : w.size(1)])
                w.clamp_(-1.0, 1.0)

        return loss.item()


def test_lr_scaling(sequences, initial_loss):
    """Test 1: Learning rate scaling."""
    print("\n" + "=" * 60)
    print("TEST 1: Learning Rate Scaling")
    print("=" * 60)
    print(f"  Initial loss: {initial_loss:.4f}")

    for lr in LR_VALUES:
        model, ckpt = load_model()
        for step in range(N_STEPS):
            x, y = sequences[step % len(sequences)]
            apply_stdp(model, x, y, lr, "lm_head")

        final_loss = compute_loss(model, sequences)
        change = (final_loss - initial_loss) / initial_loss * 100
        print(f"  LR={lr:.4f}: final={final_loss:.4f}, değişim={change:+.1f}%")


def test_multi_step(sequences, initial_loss):
    """Test 2: Multi-step learning."""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Step Learning")
    print("=" * 60)
    print(f"  Initial loss: {initial_loss:.4f}")

    for repeats in REPEAT_VALUES:
        model, ckpt = load_model()
        for step in range(N_STEPS):
            x, y = sequences[step % len(sequences)]
            for _ in range(repeats):
                apply_stdp(model, x, y, 0.001, "lm_head")

        final_loss = compute_loss(model, sequences)
        change = (final_loss - initial_loss) / initial_loss * 100
        print(f"  Repeats={repeats}: final={final_loss:.4f}, değişim={change:+.1f}%")


def test_layer_comparison(sequences, initial_loss):
    """Test 3: Layer comparison."""
    print("\n" + "=" * 60)
    print("TEST 3: Layer Karşılaştırma")
    print("=" * 60)
    print(f"  Initial loss: {initial_loss:.4f}")

    for layer in ["lm_head", "ffn"]:
        model, ckpt = load_model()
        for step in range(N_STEPS):
            x, y = sequences[step % len(sequences)]
            apply_stdp(model, x, y, 0.001, layer)

        final_loss = compute_loss(model, sequences)
        change = (final_loss - initial_loss) / initial_loss * 100
        print(f"  Layer={layer:10s}: final={final_loss:.4f}, değişim={change:+.1f}%")


def test_stdp_vs_no_stdp(sequences):
    """Test 4: STDP vs No-STDP (A/B test)."""
    print("\n" + "=" * 60)
    print("TEST 4: STDP vs No-STDP (A/B Test)")
    print("=" * 60)

    # No-STDP (baseline)
    model_no, ckpt = load_model()
    initial_no = compute_loss(model_no, sequences)
    # No update, sadece loss ölç
    final_no = initial_no

    # STDP
    model_yes, _ = load_model()
    for step in range(N_STEPS):
        x, y = sequences[step % len(sequences)]
        apply_stdp(model_yes, x, y, 0.001, "lm_head")
    final_yes = compute_loss(model_yes, sequences)

    print(f"  No-STDP:  {initial_no:.4f} → {final_no:.4f} (değişim yok)")
    print(
        f"  STDP:     {initial_no:.4f} → {final_yes:.4f} ({(final_yes - initial_no) / initial_no * 100:+.1f}%)"
    )

    if final_yes < final_no:
        print(f"\n  ✅ STDP öğreniyor (loss daha düşük)")
    else:
        print(f"\n  ❌ STDP öğrenemiyor")


def main():
    print("=" * 60)
    print("VİCDAN_SPIKE — Phase 2.6: STDP Ölçeklenebilirlik")
    print("=" * 60)

    model, ckpt = load_model()
    sequences = load_dataset(ckpt)
    initial_loss = compute_loss(model, sequences)
    print(f"\nDataset: {len(sequences)} (x, y) çifti")
    print(f"Initial loss: {initial_loss:.4f}")

    # Test 4 (en temel)
    test_stdp_vs_no_stdp(sequences)

    # Test 1
    test_lr_scaling(sequences, initial_loss)

    # Test 2
    test_multi_step(sequences, initial_loss)

    # Test 3
    test_layer_comparison(sequences, initial_loss)

    print("\n" + "=" * 60)
    print("TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()
