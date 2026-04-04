"""
VİCDAN — Phase 6.2: Task Gating + Confidence Routing

SNN input'un task'ını sınıflandırır:
  - "math" → Symbolic Engine kullan
  - "text" → SNN generate kullan

Mimari:
  Input → SNN forward → hidden state → TaskClassifier → task + confidence
  if confidence > threshold and task == "math" → Engine
  else → SNN generate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskClassifier(nn.Module):
    """
    Input hidden state'inden task sınıfı tahmin eder.

    n_tasks: sınıflar (örn: 2 = math, text)
    """

    def __init__(self, d_model, n_tasks=2):
        super().__init__()
        self.classifier = nn.Linear(d_model, n_tasks)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state: (B, T, d_model) — SNN son katman çıktısı

        Returns:
            task_logits: (B, n_tasks)
            confidence: (B,) — max softmax probability
        """
        # Sequence ortalaması (pooling)
        pooled = hidden_state.mean(dim=1)  # (B, d_model)
        logits = self.classifier(pooled)  # (B, n_tasks)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return logits, confidence
