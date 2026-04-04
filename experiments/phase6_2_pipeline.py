#!/usr/bin/env python3
"""
VİCDAN — Hybrid v0.2: Full Pipeline

Task Gating + Intent Extraction + Symbolic Engine + SNN Generate

Kullanım:
    "3+4="      → math → engine → "7"
    "merhaba"   → text → SNN generate → "merhaba nasılsın..."
    "5*8="      → math → engine → "40"
    "istanbul"  → text → SNN generate → "istanbul güzel..."
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from src.model import SpikingLM
from src.task_gating import TaskClassifier
from src.intent import extract_intent
from src.engine import run_engine

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 32
CONFIDENCE_THRESHOLD = 0.7

TASKS = ["math", "text"]


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


def load_task_classifier(d_model, checkpoint_path=None):
    classifier = TaskClassifier(d_model, n_tasks=2).to(DEVICE)
    if checkpoint_path and os.path.exists(checkpoint_path):
        classifier.load_state_dict(
            torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        )
    return classifier


def encode(text, char2idx):
    return [char2idx.get(c, 0) for c in text if c in char2idx]


def idx2char_map(raw):
    return {str(k): v for k, v in raw.items()}


class VicdanPipeline:
    """
    Hybrid Neuro-Symbolic Pipeline.

    Input → Task Gating → (math → Engine) | (text → SNN Generate)
    """

    def __init__(
        self,
        model,
        classifier,
        char2idx,
        idx2char,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    ):
        self.model = model
        self.classifier = classifier
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.confidence_threshold = confidence_threshold

    def classify_task(self, text: str) -> tuple[str, float]:
        """Input'un task'ını sınıflandır."""
        tokens = encode(text, self.char2idx)
        idx = torch.tensor([tokens], device=DEVICE)

        with torch.no_grad():
            self.model.eval()
            logits, _ = self.model(idx[:, -BLOCK_SIZE:])
            hidden = self.model.get_last_hidden()

            if hidden is None:
                return "text", 0.0

            task_logits, confidence = self.classifier(hidden)
            task_idx = task_logits.argmax(dim=-1).item()
            conf = confidence.item()

        return TASKS[task_idx], conf

    def process_math(self, text: str) -> str:
        """Math input → Intent → Engine → Result."""
        intent = extract_intent(text)
        if intent is None:
            return f"[math parse error: {text!r}]"

        result = run_engine(intent)
        if result is None:
            return f"[engine error: {text!r}]"

        return str(result)

    def process_text(self, text: str, max_tokens: int = 10) -> str:
        """Text input → SNN Generate."""
        tokens = encode(text, self.char2idx)
        if not tokens:
            return ""

        idx = torch.tensor([tokens], device=DEVICE)

        with torch.no_grad():
            self.model.eval()
            generated = self.model.generate(idx, max_new_tokens=max_tokens)
            result_chars = []
            for t in range(len(tokens), generated.shape[1]):
                c = self.idx2char.get(str(generated[0, t].item()), "?")
                if c == "\n":
                    break
                result_chars.append(c)

        return text + "".join(result_chars)

    def process(self, text: str, max_tokens: int = 10) -> dict:
        """
        Full pipeline: task gating → routing → output.

        Returns:
            {
                "input": str,
                "task": str,
                "confidence": float,
                "output": str,
                "route": str  ("engine" | "snn")
            }
        """
        task, conf = self.classify_task(text)

        if task == "math" and conf > self.confidence_threshold:
            output = self.process_math(text)
            route = "engine"
        else:
            output = self.process_text(text, max_tokens=max_tokens)
            route = "snn"

        return {
            "input": text,
            "task": task,
            "confidence": conf,
            "output": output,
            "route": route,
        }


def main():
    print("=" * 70)
    print("VİCDAN — Hybrid v0.2: Full Pipeline")
    print("=" * 70)

    vocab_size, char2idx, idx2char_raw = load_vocab()
    idx2char = idx2char_map(idx2char_raw)

    model, ckpt = load_model()
    classifier_path = os.path.join(CKPT_DIR, "task_classifier_v1.pt")
    classifier = load_task_classifier(ckpt["config"]["d_model"], classifier_path)
    print(
        f"  TaskClassifier checkpoint: {classifier_path} {'yüklendi' if os.path.exists(classifier_path) else 'BULUNAMADI'}"
    )

    pipeline = VicdanPipeline(model, classifier, char2idx, idx2char)

    print(f"\n  Model: SpikingLM (76K params)")
    print(f"  TaskClassifier: 130 params")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")

    # ── Test Inputs ──
    test_inputs = [
        # Math
        "3+4=",
        "5-2=",
        "6*7=",
        "12+34=",
        "100-50=",
        "9*9=",
        # Text
        "merhaba",
        "istanbul",
        "bir gün",
        "ahmet",
        # Edge cases
        "3+4 kaç eder?",
        "bugün hava",
        "selam nasılsın",
    ]

    print(f"\n{'=' * 70}")
    print("PIPELINE TEST")
    print(f"{'=' * 70}")

    for inp in test_inputs:
        result = pipeline.process(inp, max_tokens=8)
        route_icon = "⚙️" if result["route"] == "engine" else "🧠"
        print(f"\n  {route_icon} {result['input']!r}")
        print(f"     task={result['task']!r}, conf={result['confidence']:.3f}")
        print(f"     → {result['output']!r}")

    # ── Interactive Mode ──
    print(f"\n{'=' * 70}")
    print("INTERACTIVE MODE (çıkmak için 'q' yazın)")
    print(f"{'=' * 70}")

    try:
        while True:
            user_input = input("\n  > ").strip()
            if not user_input or user_input.lower() == "q":
                break

            result = pipeline.process(user_input, max_tokens=10)
            route_icon = "⚙️" if result["route"] == "engine" else "🧠"
            print(
                f"  {route_icon} task={result['task']!r}, conf={result['confidence']:.3f}"
            )
            print(f"  → {result['output']}")
    except (EOFError, KeyboardInterrupt):
        pass

    print("\n  Pipeline tamamlandı.")


if __name__ == "__main__":
    main()
