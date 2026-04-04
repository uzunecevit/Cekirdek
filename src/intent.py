"""
VİCDAN — Hybrid v0.1: Intent Extractor

Minimal string parsing ile matematik ifadesinden niyet çıkarımı.

Desteklenen formatlar:
    "3+4=" → expression
    "5-3=2 doğru" → statement (doğrulama)
"""

import re


def extract_intent(text: str) -> dict | None:
    """
    Matematik ifadesinden intent çıkar.

    Args:
        text: Matematik ifadesi (string)

    Returns:
        {"op": str, "a": int, "b": int} veya
        {"op": str, "a": int, "b": int, "expected": int} (statement)
    """
    text = text.strip()

    # ── Statement formatı: "5-3=2 doğru" ──
    # Regex: (sayı) (operatör) (sayı) = (sayı) [opsiyonel metin]
    stmt_match = re.search(r"(\d+)\s*([+\-*])\s*(\d+)\s*=\s*(\d+)", text)
    if stmt_match:
        a = int(stmt_match.group(1))
        op_char = stmt_match.group(2)
        b = int(stmt_match.group(3))
        expected = int(stmt_match.group(4))

        op_map = {"+": "ADD", "-": "SUB", "*": "MUL"}
        op = op_map.get(op_char)
        if op is None:
            return None

        return {"op": op, "a": a, "b": b, "expected": expected}

    # ── Expression formatı: "3+4=" ──
    if not text.endswith("="):
        text = text + "="

    if "+" in text and text.index("+") > 0:
        op = "ADD"
        sep = "+"
    elif "-" in text and text.index("-") > 0:
        op = "SUB"
        sep = "-"
    elif "*" in text and text.index("*") > 0:
        op = "MUL"
        sep = "*"
    else:
        return None

    try:
        parts = text.split(sep)
        a_str = parts[0].strip()
        b_str = parts[1].replace("=", "").strip()

        if not a_str or not b_str:
            return None

        a = int(a_str)
        b = int(b_str)
    except (ValueError, IndexError):
        return None

    return {"op": op, "a": a, "b": b}
