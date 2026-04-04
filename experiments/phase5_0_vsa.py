#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 5.0: Neuro-Symbolic VSA Test

Hedef: VSA binding ile matematiksel işlemleri vektör dönüşümü olarak modellemek.
Test: "3" ⊛ "4" ⊛ "+" → "7" benzerliği
"""

import os
import sys
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.vsa import VSA

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_vsa_binding():
    """VSA binding operatörünü test et."""
    print("=" * 70)
    print("TEST 1: VSA Binding Operatörü")
    print("=" * 70)

    vsa = VSA(n_dims=512, device=DEVICE)
    vsa.generate_math_vocab()

    # Test: 3 + 4 = 7
    three = vsa.vocab["3"]
    four = vsa.vocab["4"]
    seven = vsa.vocab["7"]
    add = vsa.vocab["+"]

    # Binding: 3 ⊛ 4 ⊛ +
    bound = vsa.bind(three, four)
    bound = vsa.bind(bound, add)

    # Cleanup: En yakın sembolü bul
    result, sim, prob = vsa.cleanup(bound)

    print(f"  3 ⊛ 4 ⊛ + → '{result}' (sim={sim:.4f}, prob={prob:.4f})")
    print(f"  7 ile benzerlik: {vsa.similarity(bound, seven):.4f}")

    # Test: 5 + 2 = 7
    five = vsa.vocab["5"]
    two = vsa.vocab["2"]
    bound2 = vsa.bind(five, two)
    bound2 = vsa.bind(bound2, add)
    result2, sim2, prob2 = vsa.cleanup(bound2)

    print(f"  5 ⊛ 2 ⊛ + → '{result2}' (sim={sim2:.4f}, prob={prob2:.4f})")
    print(f"  7 ile benzerlik: {vsa.similarity(bound2, seven):.4f}")

    # Test: 3 * 4 = 12 (two digits)
    mul = vsa.vocab["*"]
    bound3 = vsa.bind(three, four)
    bound3 = vsa.bind(bound3, mul)
    result3, sim3, prob3 = vsa.cleanup(bound3)

    print(f"  3 ⊛ 4 ⊛ * → '{result3}' (sim={sim3:.4f}, prob={prob3:.4f})")

    return vsa


def test_vsa_bundling():
    """VSA bundling operatörünü test et."""
    print(f"\n{'=' * 70}")
    print("TEST 2: VSA Bundling (Süperpozisyon)")
    print("=" * 70)

    vsa = VSA(n_dims=512, device=DEVICE)
    vsa.generate_math_vocab()

    # Bundle: 3, 4, + sembollerini bir araya getir
    three = vsa.vocab["3"]
    four = vsa.vocab["4"]
    add = vsa.vocab["+"]

    bundled = vsa.bundle([three, four, add])

    # Cleanup
    result, sim, prob = vsa.cleanup(bundled)
    print(f"  Bundle(3, 4, +) → '{result}' (sim={sim:.4f}, prob={prob:.4f})")

    # Benzerlikler
    print(f"  3 ile benzerlik: {vsa.similarity(bundled, three):.4f}")
    print(f"  4 ile benzerlik: {vsa.similarity(bundled, four):.4f}")
    print(f"  + ile benzerlik: {vsa.similarity(bundled, add):.4f}")

    return vsa


def test_vsa_sequence():
    """VSA ile sekans temsili."""
    print(f"\n{'=' * 70}")
    print("TEST 3: VSA Sekans Temsili (3+4=7)")
    print("=" * 70)

    vsa = VSA(n_dims=512, device=DEVICE)
    vsa.generate_math_vocab()

    # Sekans: 3, +, 4, =, 7
    seq = ["3", "+", "4", "=", "7"]
    vectors = [vsa.vocab[s] for s in seq]

    # Positional encoding: Her pozisyon için farklı bir vektör ile bind
    positions = [vsa.random_vector() for _ in range(len(seq))]
    encoded = [vsa.bind(v, p) for v, p in zip(vectors, positions)]

    # Bundle tüm sekansı
    seq_vector = vsa.bundle(encoded)

    # Cleanup: Her pozisyonu decode et
    print(f"  Sekans: {''.join(seq)}")
    print(f"  Sekans vektör norm: {seq_vector.norm().item():.4f}")

    for i, pos_vec in enumerate(positions):
        # Decode: seq_vector ⊛ pos_vec (binding'in tersi)
        decoded = vsa.bind(seq_vector, pos_vec)
        result, sim, prob = vsa.cleanup(decoded)
        expected = seq[i]
        match = "✅" if result == expected else "❌"
        print(
            f"  Pozisyon {i}: '{result}' (beklenen: '{expected}') {match} sim={sim:.4f}"
        )

    return vsa


def test_vsa_math_generalization():
    """VSA ile matematik genelleme testi."""
    print(f"\n{'=' * 70}")
    print("TEST 4: VSA Matematik Genelleme")
    print("=" * 70)

    vsa = VSA(n_dims=512, device=DEVICE)
    vsa.generate_math_vocab()

    # Eğitim: 3+4=7, 5+2=7, 1+6=7 gibi örneklerle "toplama=7" ilişkisi öğren
    # Test: 2+5=? → 7'yi bulabilmeli

    # Basit yaklaşım: Tüm "x+y=7" örneklerinin ortalaması
    seven = vsa.vocab["7"]
    add = vsa.vocab["+"]
    eq = vsa.vocab["="]

    examples = [("3", "4"), ("4", "3"), ("5", "2"), ("2", "5"), ("1", "6"), ("6", "1")]

    # Her örnek için: a ⊛ b ⊛ + ⊛ = → 7 benzerliği
    print("  Eğitim örnekleri (a ⊛ b ⊛ + ⊛ = → 7 benzerliği):")
    similarities = []
    for a, b in examples:
        va = vsa.vocab[a]
        vb = vsa.vocab[b]
        bound = vsa.bind(va, vb)
        bound = vsa.bind(bound, add)
        bound = vsa.bind(bound, eq)
        sim = vsa.similarity(bound, seven)
        similarities.append(sim)
        print(f"    {a}+{b}=7: sim={sim:.4f}")

    avg_sim = sum(similarities) / len(similarities)
    print(f"  Ortalama benzerlik: {avg_sim:.4f}")

    # Test: 2+5=? (eğitimde vardı ama test edelim)
    test_examples = [("2", "5"), ("3", "4"), ("8", "1")]
    print(f"\n  Test örnekleri:")
    for a, b in test_examples:
        va = vsa.vocab[a]
        vb = vsa.vocab[b]
        bound = vsa.bind(va, vb)
        bound = vsa.bind(bound, add)
        bound = vsa.bind(bound, eq)
        result, sim, prob = vsa.cleanup(bound)
        print(f"    {a}+{b}=? → '{result}' (sim={sim:.4f}, prob={prob:.4f})")

    return vsa


def main():
    print("=" * 70)
    print("VİCDAN_SPIKE — Phase 5.0: Neuro-Symbolic VSA Test")
    print("=" * 70)

    test_vsa_binding()
    test_vsa_bundling()
    test_vsa_sequence()
    test_vsa_math_generalization()

    print(f"\n{'=' * 70}")
    print("SONUÇ")
    print("=" * 70)
    print("  VSA temel operasyonlar çalışıyor.")
    print("  ⚠️ Binding tek başına sembolik mantık için yetersiz.")
    print("  ⚠️ Cleanup memory rastgele vektörlerde anlamlı sonuç vermiyor.")
    print("  → Öğrenilmiş VSA vektörleri gerekli (eğitim ile optimize edilmeli)")


if __name__ == "__main__":
    main()
