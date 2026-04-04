"""
VİCDAN — Phase 6.3: Action Head + STDP Decision Reinforcement

SNN'e ne zaman tool kullanacağını STDP ile öğretir.

Action space:
  0 = "generate"  (SNN text generation)
  1 = "use_math_engine"  (Symbolic engine)

Mimari:
  Input → SNN forward → hidden state → ActionHead → action logits
  → action seç → execute → reward
  → STDP update: ΔW = reward × error × pre^T
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    """
    Hidden state'den action seçimi yapar.

    Actions:
      0: generate (SNN text generation)
      1: use_math_engine (Symbolic engine)
    """

    def __init__(self, d_model, n_actions=2):
        super().__init__()
        self.action_head = nn.Linear(d_model, n_actions)
        self.n_actions = n_actions

        # STDP fast weights
        self.register_buffer("weight_fast", torch.zeros(n_actions, d_model))

    def forward(self, hidden_state):
        """
        Args:
            hidden_state: (B, T, d_model)

        Returns:
            action_logits: (B, n_actions)
            action_probs: (B, n_actions)
            action: (B,) — argmax
            confidence: (B,) — max probability
        """
        pooled = hidden_state.mean(dim=1)  # (B, d_model)
        W = self.action_head.weight + self.weight_fast
        logits = F.linear(pooled, W, self.action_head.bias)
        probs = F.softmax(logits, dim=-1)
        action = logits.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        return logits, probs, action, confidence

    def consolidate(self, alpha=0.05, threshold=0.01):
        """Fast weights → static weights."""
        fast_norm = self.weight_fast.norm().item()
        static_norm = self.action_head.weight.norm().item()
        importance = fast_norm / max(static_norm, 1e-8)

        if importance > threshold:
            self.action_head.weight.data = (
                1 - alpha
            ) * self.action_head.weight.data + alpha * self.weight_fast
            self.weight_fast.zero_()
            return {"status": "consolidated", "importance": importance}
        return {"status": "skipped", "importance": importance}

    def stdp_update(
        self, pre: torch.Tensor, correct_action: int, reward: float, lr: float = 0.01
    ):
        """
        STDP decision reinforcement — supervised learning with reward scaling.

        Her zaman doğru action'ı öğret. Reward, update'in büyüklüğünü belirler.
        +1 → güçlü öğrenme, -1 → zayıflatma (doğru action'ın prob'unu azaltma — bu yanlış!)

        Doğru formül:
          reward > 0 → doğru action'ı güçlendir
          reward < 0 → seçilen yanlış action'ı zayıflat (doğru action'ı değil!)
        """
        n_actions = self.n_actions

        W = self.action_head.weight + self.weight_fast
        logits = F.linear(pre.unsqueeze(0), W, self.action_head.bias)
        probs = F.softmax(logits, dim=-1)[0]

        target_onehot = torch.zeros(n_actions, device=pre.device)
        target_onehot[correct_action] = 1.0

        error = target_onehot - probs

        # Reward scaling: +1 → öğren, -1 → öğrenme yapma (skip)
        # Yanlış seçimde doğru action'ı zayıflatmak yanlış — o action zaten doğru!
        effective_lr = lr if reward > 0 else 0.0

        dw = effective_lr * error.unsqueeze(1) * pre.unsqueeze(0)
        self.weight_fast.add_(dw)
        self.weight_fast.clamp_(-10.0, 10.0)

        return {
            "reward": reward,
            "effective_lr": effective_lr,
            "dw_norm": dw.norm().item(),
            "correct_action": correct_action,
            "probs": probs.tolist(),
        }
