#!/usr/bin/env python3
"""
VİCDAN — Phase 6.6: Saliency Hypothesis Test

Test: Operatör ve keyword boost, ambiguous accuracy'yi artırır mı?

3 test:
1. A/B: Saliency ile ve olmadan → hidden state karşılaştırması
2. Sweep: Boost değerleri [1.5, 2.0, 3.0, 5.0] → ambiguous accuracy
3. Representation: Math vs text hidden state'leri daha mı ayrık?
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from src.model import SpikingLM
from src.action_head import ActionHead

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 32
ACTIONS = ["generate", "use_math_engine"]


def load_vocab():
    with open(os.path.join(DATA_DIR, "vocab.json")) as f:
        vocab = json.load(f)
    return vocab["vocab_size"], vocab["char2idx"], vocab["idx2char"]


def load_model():
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
    return model, ckpt


def encode(text, char2idx):
    return [char2idx.get(c, 0) for c in text if c in char2idx]


def compute_saliency(
    text: str, operator_boost: float = 3.0, keyword_boost: float = 2.0
) -> list[float]:
    """
    Saliency maskesi üret.

    Operatörler (+, -, *) → operator_boost
    Keywords (kaç, hesapla, sonuç) → keyword_boost
    Diğerleri → 1.0
    """
    saliency = [1.0] * len(text)

    keywords = ["kaç", "hesapla", "sonuç", "nedir", "topla", "çarp"]

    for i, ch in enumerate(text):
        if ch in ["+", "-", "*"]:
            saliency[i] = operator_boost

    for kw in keywords:
        start = 0
        while True:
            idx = text.find(kw, start)
            if idx == -1:
                break
            for j in range(idx, min(idx + len(kw), len(text))):
                saliency[j] = max(saliency[j], keyword_boost)
            start = idx + 1

    return saliency


def apply_saliency_to_hidden(
    model, text: str, saliency: list[float], char2idx: dict
) -> torch.Tensor:
    """
    Saliency'yi hidden state'e uygula.

    Basit yaklaşım: Her token'ın hidden state contribution'ını saliency ile çarp.
    Token-level saliency: input'taki her token'ın saliency değerini al,
    hidden state'in o pozisyonuna uygula.
    """
    model.eval()
    tokens = encode(text, char2idx)
    idx = torch.tensor([tokens], device=DEVICE)

    with torch.no_grad():
        logits, _ = model(idx[:, -BLOCK_SIZE:])
        hidden = model.get_last_hidden()  # (1, T, d_model)

    if hidden is None:
        return None

    # Saliency'yi token-level'a map'le
    # text'teki her karakter için saliency var, ama hidden state'te T token var
    # Basit: saliency ortalaması ile hidden state'i scale et
    saliency_tensor = torch.tensor(saliency, device=DEVICE)
    saliency_mean = saliency_tensor.mean().item()

    # Hidden state'i saliency ile modüle et
    # Her pozisyon için o pozisyondaki token'ın saliency'sini bul
    T = hidden.size(1)
    modulated_hidden = hidden.clone()

    for t in range(T):
        token_idx = idx[0, t].item()
        # Token'ın orijinal karakterini bul
        idx2char = {v: k for k, v in char2idx.items()}
        char = idx2char.get(token_idx, "")

        # Bu karakterin saliency'sini bul
        char_saliency = 1.0
        for i, c in enumerate(text):
            if c == char:
                char_saliency = saliency[i]
                break

        modulated_hidden[0, t] = modulated_hidden[0, t] * char_saliency

    return modulated_hidden


def predict_action_with_hidden(
    action_head, hidden: torch.Tensor, greedy: bool = True
) -> tuple:
    """Hidden state'den action tahmini."""
    action_head.eval()
    with torch.no_grad():
        pooled = hidden.mean(dim=1)
        W = action_head.action_head.weight + action_head.weight_fast
        logits = F.linear(pooled, W, action_head.action_head.bias)
        probs = F.softmax(logits, dim=-1)

        if greedy:
            action_idx = logits.argmax(dim=-1).item()
        else:
            action_idx = torch.multinomial(probs[0], 1).item()

        return ACTIONS[action_idx], probs[0, action_idx].item()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """İki vector arası cosine similarity. Pooled (mean) üzerinden."""
    a_pooled = a.mean(dim=1) if a.dim() == 3 else a  # (B, d_model)
    b_pooled = b.mean(dim=1) if b.dim() == 3 else b
    a_flat = a_pooled.flatten()
    b_flat = b_pooled.flatten()
    return (a_flat @ b_flat) / (a_flat.norm() * b_flat.norm() + 1e-8)


def main():
    print("=" * 70)
    print("VİCDAN — Phase 6.6: Saliency Hypothesis Test")
    print("=" * 70)

    vocab_size, char2idx, idx2char_raw = load_vocab()
    model, ckpt = load_model()
    d_model = ckpt["config"]["d_model"]

    # ActionHead — Phase 6.4 checkpoint'i yükle (outcome-based trained)
    action_head = ActionHead(d_model, n_actions=2).to(DEVICE)
    v2_ckpt = os.path.join(CKPT_DIR, "action_head_v2_outcome.pt")
    if os.path.exists(v2_ckpt):
        action_head.load_state_dict(
            torch.load(v2_ckpt, map_location=DEVICE, weights_only=True)
        )
        print(f"  ActionHead loaded: {v2_ckpt}")

    # Test samples
    clean_math = ["3+4=", "5-2=", "6*7="]
    clean_text = ["ahmet", "merhaba", "kedi"]
    ambiguous = [
        ("3+4 kaç eder?", "math"),
        ("sonucu hesapla: 3+4", "math"),
        ("mehmet 3+4 dedi", "text"),
        ("2+2=5 doğru mu?", "text"),
        ("hesapla 6*8", "math"),
        ("5+7'nin sonucu", "math"),
    ]

    # ═══════════════════════════════════════════════════════════
    # TEST 1: A/B — Saliency ile ve olmadan hidden state karşılaştırması
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 1: A/B — Saliency Effect on Hidden States")
    print(f"{'=' * 70}")

    for text, expected in ambiguous:
        saliency = compute_saliency(text, operator_boost=3.0, keyword_boost=2.0)

        # Normal hidden state
        tokens = encode(text, char2idx)
        idx = torch.tensor([tokens], device=DEVICE)
        with torch.no_grad():
            model.eval()
            model(idx[:, -BLOCK_SIZE:])
            normal_hidden = model.get_last_hidden()

        # Saliency-modulated hidden state
        modulated_hidden = apply_saliency_to_hidden(model, text, saliency, char2idx)

        if normal_hidden is not None and modulated_hidden is not None:
            sim = cosine_similarity(normal_hidden, modulated_hidden)

            # Action predictions
            normal_action, normal_conf = predict_action_with_hidden(
                action_head, normal_hidden
            )
            mod_action, mod_conf = predict_action_with_hidden(
                action_head, modulated_hidden
            )

            expected_action = "use_math_engine" if expected == "math" else "generate"
            normal_correct = normal_action == expected_action
            mod_correct = mod_action == expected_action

            print(f"\n  {text!r}")
            print(f"    Saliency: {saliency}")
            print(f"    Cosine similarity: {sim:.4f}")
            print(
                f"    Normal:   {normal_action!r} (conf={normal_conf:.3f}) {'✅' if normal_correct else '❌'}"
            )
            print(
                f"    Modulated: {mod_action!r} (conf={mod_conf:.3f}) {'✅' if mod_correct else '❌'}"
            )

    # ═══════════════════════════════════════════════════════════
    # TEST 2: Boost Sweep — Hangi boost değeri en iyi?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 2: Boost Sweep — Operator & Keyword Boost Values")
    print(f"{'=' * 70}")

    boost_values = [1.5, 2.0, 3.0, 5.0, 10.0]
    results = {}

    print(
        f"\n  {'Op Boost':<10} {'KW Boost':<10} {'Clean Acc':<12} {'Ambig Acc':<12} {'Avg Sim':<10}"
    )
    print(f"  {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 10}")

    for op_boost in boost_values:
        for kw_boost in [
            1.0,
            op_boost * 0.67,
            op_boost,
        ]:  # KW = op_boost'un 2/3'ü veya eşit
            clean_correct = 0
            clean_total = 0
            amb_correct = 0
            amb_total = 0
            sims = []

            for text in clean_math + clean_text:
                saliency = compute_saliency(text, op_boost, kw_boost)
                mod_hidden = apply_saliency_to_hidden(model, text, saliency, char2idx)
                if mod_hidden is None:
                    continue

                expected = "use_math_engine" if text in clean_math else "generate"
                action, _ = predict_action_with_hidden(action_head, mod_hidden)
                if action == expected:
                    clean_correct += 1
                clean_total += 1

            for text, expected_label in ambiguous:
                saliency = compute_saliency(text, op_boost, kw_boost)
                mod_hidden = apply_saliency_to_hidden(model, text, saliency, char2idx)
                if mod_hidden is None:
                    continue

                expected = "use_math_engine" if expected_label == "math" else "generate"
                action, _ = predict_action_with_hidden(action_head, mod_hidden)
                if action == expected:
                    amb_correct += 1
                amb_total += 1

            clean_acc = clean_correct / max(clean_total, 1) * 100
            amb_acc = amb_correct / max(amb_total, 1) * 100

            key = f"{op_boost:.1f}/{kw_boost:.1f}"
            results[key] = {"clean": clean_acc, "ambig": amb_acc}

            marker = " ← BEST" if amb_acc >= 66 and clean_acc >= 100 else ""
            print(
                f"  {op_boost:<10.1f} {kw_boost:<10.1f} {clean_acc:<12.0f} {amb_acc:<12.0f} {'':<10}{marker}"
            )

    # ═══════════════════════════════════════════════════════════
    # TEST 3: Representation Analysis — Math vs Text ayrımı
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 3: Representation Analysis — Math vs Text Separation")
    print(f"{'=' * 70}")

    # Hidden state'leri topla
    math_hiddens = []
    text_hiddens = []
    math_hiddens_mod = []
    text_hiddens_mod = []

    for text in clean_math:
        tokens = encode(text, char2idx)
        idx = torch.tensor([tokens], device=DEVICE)
        with torch.no_grad():
            model.eval()
            model(idx[:, -BLOCK_SIZE:])
            h = model.get_last_hidden()
        if h is not None:
            math_hiddens.append(h)
            saliency = compute_saliency(text, 3.0, 2.0)
            hm = apply_saliency_to_hidden(model, text, saliency, char2idx)
            if hm is not None:
                math_hiddens_mod.append(hm)

    for text in clean_text:
        tokens = encode(text, char2idx)
        idx = torch.tensor([tokens], device=DEVICE)
        with torch.no_grad():
            model.eval()
            model(idx[:, -BLOCK_SIZE:])
            h = model.get_last_hidden()
        if h is not None:
            text_hiddens.append(h)
            saliency = compute_saliency(text, 3.0, 2.0)
            hm = apply_saliency_to_hidden(model, text, saliency, char2idx)
            if hm is not None:
                text_hiddens_mod.append(hm)

    # Within-class ve between-class similarity
    if math_hiddens and text_hiddens:
        # Normal: math-math similarity
        mm_sims = []
        for i in range(len(math_hiddens)):
            for j in range(i + 1, len(math_hiddens)):
                mm_sims.append(cosine_similarity(math_hiddens[i], math_hiddens[j]))

        # Normal: text-text similarity
        tt_sims = []
        for i in range(len(text_hiddens)):
            for j in range(i + 1, len(text_hiddens)):
                tt_sims.append(cosine_similarity(text_hiddens[i], text_hiddens[j]))

        # Normal: math-text similarity (between-class)
        mt_sims = []
        for mh in math_hiddens:
            for th in text_hiddens:
                mt_sims.append(cosine_similarity(mh, th))

        print(f"\n  Normal Hidden States:")
        print(f"    Math-Math similarity:   {sum(mm_sims) / max(len(mm_sims), 1):.4f}")
        print(f"    Text-Text similarity:   {sum(tt_sims) / max(len(tt_sims), 1):.4f}")
        print(f"    Math-Text similarity:   {sum(mt_sims) / max(len(mt_sims), 1):.4f}")
        print(
            f"    Separation (1 - Math-Text): {1 - sum(mt_sims) / max(len(mt_sims), 1):.4f}"
        )

        # Modulated
        if math_hiddens_mod and text_hiddens_mod:
            mm_sims_mod = []
            for i in range(len(math_hiddens_mod)):
                for j in range(i + 1, len(math_hiddens_mod)):
                    mm_sims_mod.append(
                        cosine_similarity(math_hiddens_mod[i], math_hiddens_mod[j])
                    )

            tt_sims_mod = []
            for i in range(len(text_hiddens_mod)):
                for j in range(i + 1, len(text_hiddens_mod)):
                    tt_sims_mod.append(
                        cosine_similarity(text_hiddens_mod[i], text_hiddens_mod[j])
                    )

            mt_sims_mod = []
            for mh in math_hiddens_mod:
                for th in text_hiddens_mod:
                    mt_sims_mod.append(cosine_similarity(mh, th))

            print(f"\n  Saliency-Modulated Hidden States:")
            print(
                f"    Math-Math similarity:   {sum(mm_sims_mod) / max(len(mm_sims_mod), 1):.4f}"
            )
            print(
                f"    Text-Text similarity:   {sum(tt_sims_mod) / max(len(tt_sims_mod), 1):.4f}"
            )
            print(
                f"    Math-Text similarity:   {sum(mt_sims_mod) / max(len(mt_sims_mod), 1):.4f}"
            )
            print(
                f"    Separation (1 - Math-Text): {1 - sum(mt_sims_mod) / max(len(mt_sims_mod), 1):.4f}"
            )

            sep_improvement = (1 - sum(mt_sims_mod) / max(len(mt_sims_mod), 1)) - (
                1 - sum(mt_sims) / max(len(mt_sims), 1)
            )
            print(f"\n  Separation improvement: {sep_improvement:+.4f}")
            if sep_improvement > 0.05:
                print(f"  ✅ Saliency math/text ayrımını iyileştiriyor")
            else:
                print(f"  ⚠️ Saliency math/text ayrımını önemli ölçüde değiştirmiyor")

    # ── Sonuçları Kaydet ──
    results_path = os.path.join(
        os.path.dirname(__file__), "phase6_6_saliency_test.json"
    )
    with open(results_path, "w") as f:
        json.dump(
            {
                "boost_sweep": results,
                "note": "Saliency hypothesis test results",
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved: {results_path}")


if __name__ == "__main__":
    main()
