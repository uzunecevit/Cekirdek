"""
VİCDAN — Hybrid v0.1: Symbolic Engine

Deterministik matematik motoru.
SNN hesap yapmaz, bu modül yapar.
"""


def run_engine(intent: dict) -> int | None:
    """
    Intent'i al, deterministic sonuç üret.

    Args:
        intent: {"op": str, "a": int, "b": int}

    Returns:
        Sonuç (int) veya None
    """
    if intent is None:
        return None

    op = intent["op"]
    a = intent["a"]
    b = intent["b"]

    if op == "ADD":
        return a + b
    elif op == "SUB":
        return a - b
    elif op == "MUL":
        return a * b

    return None
