#!/usr/bin/env python3
"""
Çekirdek — Faz 7.0.5: Qwen3.5-9B Teacher Validation (GGUF)

Qwen'in Türkçe doğal dil girdilerinde doğru action'ı seçip seçmediğini test eder.
GGUF formatı kullanır (llama-cpp-python).
"""

import sys
import os
import json
import re
import time

from llama_cpp import Llama

# Config
MODEL_PATH = "/home/ayandon/KAPTAN/modeller/Qwen3.5-9B-Q4_K_M.gguf"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.1
N_CTX = 2048
N_GPU_LAYERS = -1  # Tüm katmanları GPU'ya yükle

# Test seti
TEST_CASES = [
    # Compute (8)
    ("3+4=", "compute"),
    ("5-2=", "compute"),
    ("6*7=", "compute"),
    ("12+34=", "compute"),
    ("100-50=", "compute"),
    ("9*9=", "compute"),
    ("7*8=", "compute"),
    ("15-7=", "compute"),
    # Verify (5)
    ("5-3=2 doğru", "verify"),
    ("4*4=16 doğru mu?", "verify"),
    ("3+7=10 doğru", "verify"),
    ("8-5=2 doğru mu?", "verify"),
    ("6*6=35 doğru mu?", "verify"),
    # Generate (8)
    ("merhaba", "generate"),
    ("istanbul", "generate"),
    ("bir gün", "generate"),
    ("ahmet", "generate"),
    ("nasılsın", "generate"),
    ("bugün hava", "generate"),
    ("kedi", "generate"),
    ("araba", "generate"),
]


def load_model():
    """Qwen3.5-9B GGUF'yi yükle."""
    print(f"  Model yükleniyor: {MODEL_PATH}")
    print(f"  GPU layers: {N_GPU_LAYERS} (tümü)")

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    print(f"  Model yüklendi.")
    return llm


def classify_action(response: str, prompt: str) -> str:
    """
    Qwen çıktısından action çıkar.

    Thinking mode'u temizle, sonra sınıflandır.
    """
    text = response.strip()

    # Thinking Process: ... (think) ...  pattern'ini temizle
    # "Thinking Process:" ile başlıyor, ardından düşünce geliyor
    # Sonrasında gerçek yanıt var
    thinking_match = re.search(r"Thinking Process:.*?(?:\n\n|\n|$)", text, re.DOTALL)
    if thinking_match:
        # Thinking Process'ten sonrasını al
        text = text[thinking_match.end() :].strip()
        # Eğer thinking sonrası boşsa, ilk 200 karakteri al
        if not text:
            text = response[200:].strip()

    # Hâlâ thinking içeriyorsa, daha agresif temizle
    if "Thinking Process" in text or "Analyze the Request" in text:
        # İlk gerçek cümleyi bul (genellikle 2. satırdan sonra)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if (
                line.strip()
                and "Thinking" not in line
                and "Analyze" not in line
                and "Identify" not in line
            ):
                text = "\n".join(lines[i:])
                break

    text = text.lower()

    # Compute: çıktı sadece sayı veya basit matematik sonucu
    clean = re.sub(r"[^\d]", "", text)
    if clean and len(clean) <= 5 and len(text.strip()) <= 10:
        try:
            int(clean)
            return "compute"
        except ValueError:
            pass

    # Verify: evet/hayır/doğru/yanlış içeriyor
    verify_keywords = ["evet", "hayır", "doğru", "yanlış", "true", "false"]
    if any(kw in text for kw in verify_keywords):
        return "verify"

    # Compute: matematiksel ifade içeriyor (örn: "3+4=7")
    if re.search(r"\d+\s*[+\-*/]\s*\d+\s*=\s*\d+", text):
        return "compute"

    # Aksi halde generate
    return "generate"


def run_test(llm):
    """Tüm test case'lerini çalıştır."""
    results = []
    total_time = 0

    print(f"\n{'=' * 70}")
    print("TEST BAŞLIYOR")
    print(f"{'=' * 70}")

    for i, (prompt, expected) in enumerate(TEST_CASES):
        start = time.time()

        # Generate — completion API (thinking mode yok)
        output = llm(
            prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            stop=["\n", "\n\n"],
        )
        response = output["choices"][0]["text"].strip()
        elapsed = time.time() - start
        total_time += elapsed

        # Classify
        predicted = classify_action(response, prompt)
        correct = predicted == expected

        results.append(
            {
                "prompt": prompt,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "response": response,
                "time": elapsed,
            }
        )

        # Progress
        status = "✅" if correct else "❌"
        print(
            f"  [{i + 1:2d}/{len(TEST_CASES)}] {status} {prompt!r:25s} → {predicted!r:10s} (beklenen: {expected!r:10s}) | {response[:40]!r}"
        )

    return results, total_time


def analyze_results(results, total_time):
    """Sonuçları analiz et."""
    print(f"\n{'=' * 70}")
    print("SONUÇLAR")
    print(f"{'=' * 70}")

    # Overall
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    overall_acc = correct / total * 100

    # Per-category
    categories = {}
    for r in results:
        cat = r["expected"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1

    print(f"\n  Overall Accuracy: {correct}/{total} ({overall_acc:.0f}%)")
    print(f"  Total Time: {total_time:.1f}s ({total_time / total:.1f}s per sample)")

    for cat, stats in categories.items():
        acc = stats["correct"] / stats["total"] * 100
        print(f"    {cat:12s}: {stats['correct']}/{stats['total']} ({acc:.0f}%)")

    # Yanlışlar
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n  Yanlışlar ({len(wrong)}):")
        for r in wrong:
            print(
                f"    ❌ {r['prompt']!r:25s} → {r['predicted']!r:10s} (beklenen: {r['expected']!r:10s}) | {r['response'][:50]!r}"
            )

    # Karar
    print(f"\n{'=' * 70}")
    print("KARAR")
    print(f"{'=' * 70}")

    if overall_acc >= 70:
        print(f"  ✅ Qwen öğretmen olarak UYGUN (overall {overall_acc:.0f}% > %70)")
        decision = "PASS"
    elif overall_acc >= 50:
        print(
            f"  ⚠️ Qwen kısmen uygun (overall {overall_acc:.0f}%, sentetik veri ile iyileştirilebilir)"
        )
        decision = "MARGINAL"
    else:
        print(
            f"  ❌ Qwen öğretmen olarak UYGUN DEĞİL (overall {overall_acc:.0f}% < %50)"
        )
        decision = "FAIL"

    return {
        "overall_accuracy": overall_acc,
        "categories": {
            cat: stats["correct"] / stats["total"] * 100
            for cat, stats in categories.items()
        },
        "correct": correct,
        "total": total,
        "decision": decision,
        "total_time": total_time,
    }


def main():
    print("=" * 70)
    print("Çekirdek — Faz 7.0.5: Qwen3.5-9B Teacher Validation (GGUF)")
    print("=" * 70)
    print(f"  Model: Qwen3.5-9B-Q4_K_M.gguf")
    print(f"  Test cases: {len(TEST_CASES)}")
    print(f"  Max tokens: {MAX_NEW_TOKENS}")
    print(f"  Temperature: {TEMPERATURE}")

    # Load model
    llm = load_model()

    # Run tests
    results, total_time = run_test(llm)

    # Analyze
    summary = analyze_results(results, total_time)

    # Save results
    output = {
        "model": "Qwen3.5-9B-Q4_K_M.gguf",
        "test_cases": len(TEST_CASES),
        "results": results,
        "summary": summary,
    }

    output_path = os.path.join(
        os.path.dirname(__file__), "phase7_0_5_qwen_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
