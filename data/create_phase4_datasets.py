#!/usr/bin/env python3
"""
VİCDAN_SPIKE — Phase 4.1 Dataset Oluşturucu
Türkçe çekim + Matematik denklemleri
"""

import os
import random

random.seed(42)
DATA_DIR = os.path.dirname(__file__)

# ──────────────────────────────────────────────────────────
# Türkçe Çekim Örnekleri
# ──────────────────────────────────────────────────────────


def generate_turkish():
    roots = {
        "ev": ["evler", "evim", "evin", "evimiz", "evde", "evden", "eve", "evi"],
        "koş": [
            "koştu",
            "koşuyor",
            "koşacak",
            "koşmalı",
            "koşmaz",
            "koşar",
            "koşma",
            "koşsun",
        ],
        "gel": [
            "geldi",
            "geliyor",
            "gelecek",
            "gelmeli",
            "gelmez",
            "gelir",
            "gelme",
            "gelsin",
        ],
        "oku": [
            "okudu",
            "okuyor",
            "okuyacak",
            "okumalı",
            "okumaz",
            "okur",
            "okuma",
            "okusun",
        ],
        "yaz": [
            "yazdı",
            "yazıyor",
            "yazacak",
            "yazmalı",
            "yazmaz",
            "yazar",
            "yazma",
            "yazsın",
        ],
        "git": [
            "gitti",
            "gidiyor",
            "gidecek",
            "gitmeli",
            "gitmez",
            "gider",
            "gitme",
            "gitsin",
        ],
        "yap": [
            "yaptı",
            "yapıyor",
            "yapacak",
            "yapmalı",
            "yapmaz",
            "yapar",
            "yapma",
            "yapsın",
        ],
        "al": ["aldı", "alıyor", "alacak", "almalı", "almaz", "alır", "alma", "alsın"],
        "ver": [
            "verdi",
            "veriyor",
            "verecek",
            "vermeli",
            "vermez",
            "verir",
            "verme",
            "versin",
        ],
        "gör": [
            "gördü",
            "görüyor",
            "görecek",
            "görmeli",
            "görmez",
            "görür",
            "görme",
            "görsün",
        ],
        "bil": [
            "bildi",
            "biliyor",
            "bilecek",
            "bilmeli",
            "bilmez",
            "bilir",
            "bilme",
            "bilsin",
        ],
        "çalış": [
            "çalıştı",
            "çalışıyor",
            "çalışacak",
            "çalışmalı",
            "çalışmaz",
            "çalışır",
        ],
        "öğren": [
            "öğrendi",
            "öğreniyor",
            "öğrenecek",
            "öğrenmeli",
            "öğrenmez",
            "öğrenir",
        ],
        "düşün": [
            "düşündü",
            "düşünüyor",
            "düşünecek",
            "düşünmeli",
            "düşünmez",
            "düşünür",
        ],
        "konuş": [
            "konuştu",
            "konuşuyor",
            "konuşacak",
            "konuşmalı",
            "konuşmaz",
            "konuşur",
        ],
    }

    samples = []
    for root, forms in roots.items():
        for form in forms:
            samples.append(f"{root}>{form}")

    # Atasözleri / kısa cümleler
    sentences = [
        "damlaya damlaya göl olur",
        "acele işe şeytan karışır",
        "güzel söz insanın aynasıdır",
        "bugünün işini yarına bırakma",
        "ata binen nalını mıhını arar",
        "sakla samanı gelir zamanı",
        "göz görür gönül çeker",
        "iyi söz insanı evinden eder",
        "köprüyü geçinceye kadar ayıya dayı de",
        "damlaya damlaya göl olur",
    ]
    for s in sentences:
        samples.append(f">>{s}")

    path = os.path.join(DATA_DIR, "turkish.txt")
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s + "\n")

    print(f"Türkçe: {len(samples)} örnek → {path}")
    return samples


# ──────────────────────────────────────────────────────────
# Matematik Denklemleri
# ──────────────────────────────────────────────────────────


def generate_math():
    equations = []

    # Toplama
    for _ in range(150):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        equations.append(f"{a}+{b}={a + b}")

    # Çıkarma
    for _ in range(100):
        a = random.randint(2, 20)
        b = random.randint(1, a - 1)
        equations.append(f"{a}-{b}={a - b}")

    # Çarpma
    for _ in range(50):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        equations.append(f"{a}*{b}={a * b}")

    path = os.path.join(DATA_DIR, "math.txt")
    with open(path, "w", encoding="utf-8") as f:
        for eq in equations:
            f.write(eq + "\n")

    print(f"Matematik: {len(equations)} örnek → {path}")
    return equations


if __name__ == "__main__":
    generate_turkish()
    generate_math()
