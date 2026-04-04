"""
VİCDAN — Hybrid v0.1: Intent + Engine Bağımsız Test

SNN yok. Sadece string işleme ve matematik doğrulaması.
"""

import sys
import os

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, src_dir)

from intent import extract_intent
from engine import run_engine

test_cases = [
    ("3+4=", 7),
    ("5-2=", 3),
    ("6*7=", 42),
    ("0+0=", 0),
    ("9-9=", 0),
    ("1*1=", 1),
    ("12+34=", 46),
    ("100-50=", 50),
    ("8*8=", 64),
]

print("=" * 50)
print("HYBRID v0.1 — Intent + Engine Test")
print("=" * 50)

passed = 0
failed = 0

for expr, expected in test_cases:
    intent = extract_intent(expr)
    if intent is None:
        print(f"  FAIL: {expr!r} → intent None")
        failed += 1
        continue

    result = run_engine(intent)
    status = "PASS" if result == expected else "FAIL"

    if status == "PASS":
        passed += 1
    else:
        failed += 1

    print(
        f"  {status}: {expr!r} → intent={intent}, result={result}, expected={expected}"
    )

print("=" * 50)
print(f"Sonuç: {passed} passed, {failed} failed")
print("=" * 50)

if failed == 0:
    print("\nAdım 1 tamamlandı. Intent + Engine çalışıyor.")
else:
    print(f"\n{failed} test başarısız!")
    sys.exit(1)
