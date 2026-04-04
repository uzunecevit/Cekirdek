"""
VİCDAN_SPIKE — Vocab ve Dataset Oluşturucu

50 karakterlik yeni vocab:
- a-z (26)
- Türkçe: ç, ğ, ı, ö, ş, ü, Ç, Ğ, İ, Ö, Ş, Ü (12)
- Rakamlar: 0-9 (10)
- Operatörler: +, -, *, =, /, ., ,, (, ), :, ;, !, ? (12)
- Özel: BOS, SPACE, \n (3)

Toplam: ~63 karakter
"""

import os
import json
import random

random.seed(42)

# ──────────────────────────────────────────────────────────
# Vocab Tanımı
# ──────────────────────────────────────────────────────────

CHARS = "abcdefghijklmnopqrstuvwxyzçğıöşüÇĞİÖŞÜ0123456789+-*/=.,():;!? \n"

BOS_TOKEN = "<BOS>"
VOCAB = [BOS_TOKEN] + list(CHARS)
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

print(f"Vocab boyutu: {VOCAB_SIZE}")
print(f"Karakterler: {''.join(VOCAB[:30])}...")

# ──────────────────────────────────────────────────────────
# Dataset Oluşturucu
# ──────────────────────────────────────────────────────────


def generate_pretrain_dataset():
    """İsimler + Türkçe ekler + Matematik karışımı."""
    samples = []

    # 1. İsimler (İngilizce)
    names_path = os.path.join(
        os.path.dirname(__file__), "../../VİCDAN_HEBBIAN/micro/names.txt"
    )
    if os.path.exists(names_path):
        with open(names_path) as f:
            names = [line.strip() for line in f if line.strip()]
        random.shuffle(names)
        for name in names[:200]:
            if all(c in CHAR2IDX for c in name.lower()):
                samples.append(name.lower())

    # 2. Türkçe ek yapıları
    turkish_suffixes = [
        "koştu",
        "koştular",
        "koşacak",
        "koşmalı",
        "koşuyor",
        "geldi",
        "geldiler",
        "gelecek",
        "gelmeli",
        "geliyor",
        "evler",
        "evlerimiz",
        "evlerimizden",
        "evlerimizdeki",
        "güzel",
        "güzeller",
        "güzelleşti",
        "güzelleşecek",
        "çalıştı",
        "çalışıyor",
        "çalışacak",
        "çalışmalı",
        "yaptı",
        "yapıyor",
        "yapacak",
        "yapmalı",
        "büyük",
        "küçük",
        "uzun",
        "kısa",
    ]
    samples.extend(turkish_suffixes)

    # 3. Türkçe atasözleri / kısa cümleler
    turkish_sentences = [
        "damlaya damlaya göl olur",
        "acele işe şeytan karışır",
        "güzel söz insanın aynasıdır",
        "bilim akıldan olur akıllıdan değil",
        "ata binen nalını mıhını arar",
        "bugünün işini yarına bırakma",
    ]
    samples.extend(turkish_sentences)

    # 4. Basit matematik
    math_samples = []
    for _ in range(100):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            result = a + b
        elif op == "-":
            if a < b:
                a, b = b, a
            result = a - b
        else:
            result = a * b
        math_samples.append(f"{a}{op}{b}={result}")
    samples.extend(math_samples)

    # Filtre: Sadece vocab'daki karakterleri içerenler
    valid_samples = []
    for s in samples:
        if all(c in CHAR2IDX for c in s):
            valid_samples.append(s)

    print(f"Toplam sample: {len(valid_samples)}")
    print(f"  İsimler: {sum(1 for s in valid_samples if s.isalpha())}")
    print(f"  Matematik: {sum(1 for s in valid_samples if '=' in s)}")
    print(
        f"  Türkçe: {sum(1 for s in valid_samples if any(c in s for c in 'çğıöşüÇĞİÖŞÜ'))}"
    )

    return valid_samples


if __name__ == "__main__":
    samples = generate_pretrain_dataset()

    # Kaydet
    data_dir = os.path.join(os.path.dirname(__file__))
    with open(
        os.path.join(data_dir, "pretrain_samples.txt"), "w", encoding="utf-8"
    ) as f:
        for s in samples:
            f.write(s + "\n")

    # Vocab kaydet
    vocab_info = {
        "vocab_size": VOCAB_SIZE,
        "chars": VOCAB,
        "char2idx": CHAR2IDX,
        "idx2char": IDX2CHAR,
    }
    with open(os.path.join(data_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, ensure_ascii=False, indent=2)

    print(f"\nKaydedildi:")
    print(f"  {os.path.join(data_dir, 'pretrain_samples.txt')} ({len(samples)} satır)")
    print(f"  {os.path.join(data_dir, 'vocab.json')}")
