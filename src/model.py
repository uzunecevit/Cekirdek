"""
VİCDAN_SPIKE — Minimal Spiking Neural Network
Nord tarzı SNN: LIF nöronlar, surrogate gradient, online STDP öğrenme.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────────────────
# Surrogate Gradient
# ──────────────────────────────────────────────────────────


class TernarySpikeFunction(torch.autograd.Function):
    """
    Ternary spike: {-1, 0, +1}
    Forward: v > +thr → +1, v < -thr → -1, else 0
    Backward: triangle surrogate gradient
    """

    @staticmethod
    def forward(ctx, v, threshold):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        pos = (v > threshold).float()
        neg = (v < -threshold).float()
        return pos - neg

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        threshold = ctx.threshold
        # Triangle surrogate: max(0, alpha - |v|)
        alpha = 2.0
        grad = grad_output * torch.clamp(alpha - (v.abs() - threshold).abs(), 0, alpha)
        return grad, None


class SpikeFunction(torch.autograd.Function):
    """
    Forward: step function (V > threshold → 1, else 0)
    Backward: sigmoid gradient (smooth approximation)
    """

    @staticmethod
    def forward(ctx, v, threshold):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        return (v > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        threshold = ctx.threshold
        alpha = 5.0
        sig = torch.sigmoid(alpha * (v - threshold))
        grad = grad_output * alpha * sig * (1 - sig)
        return grad, None


# ──────────────────────────────────────────────────────────
# LIF Neuron (Leaky Integrate-and-Fire)
# ──────────────────────────────────────────────────────────


class LIFNeuron(nn.Module):
    """
    Ternary LIF nöron: {-1, 0, +1} spike + amplitude encoding.

    Binary {0,1} → Ternary {-1,0,+1} (SpikeLM yaklaşımı)
    Amplitude: spike * alpha (layerwise)

    V_t = decay * V_{t-1} + input
    spike = +1 if V > +thr, -1 if V < -thr, else 0
    V_t = V_t - spike * threshold  (soft reset)
    """

    def __init__(
        self,
        threshold=1.0,
        decay=0.3,
        amplitude=1.0,
        surrogate_alpha=5.0,
        use_surrogate=False,
        ternary=True,
    ):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.amplitude = amplitude  # Spike amplitude (alpha)
        self.surrogate_alpha = surrogate_alpha
        self.use_surrogate = use_surrogate
        self.ternary = ternary  # True = {-1,0,+1}, False = {0,1}
        self.v = None
        self.spike_count = 0
        self.total_steps = 0
        self.pos_count = 0  # +1 spike sayısı
        self.neg_count = 0  # -1 spike sayısı

    def reset(self, batch_size, hidden_dim, device):
        self.v = torch.zeros(batch_size, hidden_dim, device=device)
        self.spike_count = 0
        self.total_steps = 0
        self.pos_count = 0
        self.neg_count = 0

    def forward(self, x):
        # Membrane potential update
        self.v = self.decay * self.v + x

        if self.ternary:
            # Ternary spike: {-1, 0, +1}
            if self.use_surrogate:
                spike = TernarySpikeFunction.apply(self.v, self.threshold)
            else:
                with torch.no_grad():
                    spike = torch.where(
                        self.v > self.threshold,
                        torch.tensor(1.0, device=self.v.device),
                        torch.where(
                            self.v < -self.threshold,
                            torch.tensor(-1.0, device=self.v.device),
                            torch.tensor(0.0, device=self.v.device),
                        ),
                    )
        else:
            # Binary spike: {0, 1}
            if self.use_surrogate:
                spike = SpikeFunction.apply(self.v, self.threshold)
            else:
                with torch.no_grad():
                    spike = (self.v > self.threshold).float()

        # Amplitude scaling
        spike = spike * self.amplitude

        # Soft reset: spike yönünde potansiyeli azalt
        self.v = self.v - spike.detach()

        # Stats
        if self.ternary:
            self.pos_count += (spike > 0).sum().item()
            self.neg_count += (spike < 0).sum().item()
        self.spike_count += spike.abs().sum().item()
        self.total_steps += spike.numel()

        return spike

    @property
    def spike_rate(self):
        if self.total_steps == 0:
            return 0.0
        return self.spike_count / self.total_steps

    @property
    def balance_ratio(self):
        """+1 / -1 oranı. 1.0 = dengeli, >1 = pozitif bias."""
        if self.neg_count == 0:
            return float("inf") if self.pos_count > 0 else 1.0
        return self.pos_count / self.neg_count


# ──────────────────────────────────────────────────────────
# Spiking Linear (weight → LIF)
# ──────────────────────────────────────────────────────────


class SpikingLinear(nn.Module):
    """
    Fast weight destekli Spiking Linear katmanı.

    W_eff = W_static + W_fast
    W_fast: STDP ile güncellenir (geçici)
    consolidate(): W_fast → W_static sızdırma (kalıcı)
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        threshold=1.0,
        decay=0.3,
        amplitude=1.0,
        use_surrogate=False,
        ternary=True,
        fast_weight=False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.lif = LIFNeuron(
            threshold=threshold,
            decay=decay,
            amplitude=amplitude,
            use_surrogate=use_surrogate,
            ternary=ternary,
        )
        self.fast_weight = fast_weight
        if fast_weight:
            self.register_buffer("weight_fast", torch.zeros(out_dim, in_dim))
            self.consolidation_count = 0

    def reset(self, batch_size, device):
        self.lif.reset(batch_size, self.linear.out_features, device)

    def get_weight(self):
        if self.fast_weight:
            return self.linear.weight + self.weight_fast
        return self.linear.weight

    def forward(self, x):
        W = self.get_weight()
        return self.lif(F.linear(x, W))

    def consolidate(self, alpha=0.05):
        """W_fast → W_static sızdırma."""
        if not self.fast_weight:
            return {"status": "no_fast_weight"}

        fast_norm = self.weight_fast.norm().item()
        static_norm = self.linear.weight.norm().item()
        importance = fast_norm / max(static_norm, 1e-8)

        if importance > 0.01:  # Minimum eşik
            self.linear.weight.data = (
                1 - alpha
            ) * self.linear.weight.data + alpha * self.weight_fast
            self.weight_fast.zero_()
            self.consolidation_count += 1

            return {
                "status": "consolidated",
                "alpha": alpha,
                "fast_norm_before": fast_norm,
                "static_norm_after": self.linear.weight.norm().item(),
                "importance": importance,
            }
        return {"status": "skipped", "importance": importance}


# ──────────────────────────────────────────────────────────
# Spiking Attention (basitleştirilmiş)
# ──────────────────────────────────────────────────────────


class SpikingSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_head=4,
        threshold=1.0,
        decay=0.3,
        amplitude=1.0,
        use_surrogate=False,
        ternary=True,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.wq = SpikingLinear(
            d_model, d_model, threshold, decay, amplitude, use_surrogate, ternary
        )
        self.wk = SpikingLinear(
            d_model, d_model, threshold, decay, amplitude, use_surrogate, ternary
        )
        self.wv = SpikingLinear(
            d_model, d_model, threshold, decay, amplitude, use_surrogate, ternary
        )
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def reset(self, batch_size, device):
        self.wq.reset(batch_size, device)
        self.wk.reset(batch_size, device)
        self.wv.reset(batch_size, device)

    def forward(self, x):
        # x: (B, C) — tek token
        if x.dim() == 2:
            B, C = x.size()
            T = 1
            q = self.wq(x)  # (B, C)
            k = self.wk(x)
            v = self.wv(x)
            return self.wo(x)  # (B, C)
        else:
            B, T, C = x.size()
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.wo(out)


# ──────────────────────────────────────────────────────────
# Spiking FFN
# ──────────────────────────────────────────────────────────


class SpikingFFN(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        threshold=1.0,
        decay=0.3,
        amplitude=1.0,
        use_surrogate=False,
        ternary=True,
        fast_weight_fc2=False,
    ):
        super().__init__()
        self.fc1 = SpikingLinear(
            d_model, d_ff, threshold, decay, amplitude, use_surrogate, ternary
        )
        self.fc2 = SpikingLinear(
            d_ff,
            d_model,
            threshold,
            decay,
            amplitude,
            use_surrogate,
            ternary,
            fast_weight=fast_weight_fc2,
        )

    def reset(self, batch_size, device):
        self.fc1.reset(batch_size, device)
        self.fc2.reset(batch_size, device)
        self._pre_activation = None
        self._post_activation = None

    def forward(self, x):
        h = self.fc1(x)
        self._pre_activation = h  # fc1 çıkışı (fc2 girdisi)
        out = self.fc2(h)
        self._post_activation = out  # fc2 çıkışı
        if x.dim() == 3 and x.size(1) == 1:
            return out.unsqueeze(1)
        return out


# ──────────────────────────────────────────────────────────
# Spiking Transformer Block
# ──────────────────────────────────────────────────────────


class SpikingTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        d_ff,
        threshold=1.0,
        decay=0.3,
        amplitude=1.0,
        use_surrogate=False,
        ternary=True,
        fast_weight_fc2=False,
    ):
        super().__init__()
        self.attn = SpikingSelfAttention(
            d_model, n_head, threshold, decay, amplitude, use_surrogate, ternary
        )
        self.ffn = SpikingFFN(
            d_model,
            d_ff,
            threshold,
            decay,
            amplitude,
            use_surrogate,
            ternary,
            fast_weight_fc2=fast_weight_fc2,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def reset(self, batch_size, device):
        self.attn.reset(batch_size, device)
        self.ffn.reset(batch_size, device)

    def forward(self, x):
        # x: (B, C) — tek token
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────
# SpikingLM (Ana Model)
# ──────────────────────────────────────────────────────────


class SpikingLM(nn.Module):
    """
    Nord tarzı Spiking Language Model.
    Embedding → Spiking Transformer Blocks → LM Head
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_layer=4,
        n_head=4,
        d_ff=256,
        block_size=64,
        threshold=1.0,
        decay=0.3,
        amplitude=1.0,
        use_surrogate=False,
        ternary=True,
        fast_weight_fc2=False,
    ):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model

        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model)
        self.input_lif = LIFNeuron(
            threshold=threshold,
            decay=decay,
            amplitude=amplitude,
            use_surrogate=use_surrogate,
            ternary=ternary,
        )

        self.blocks = nn.ModuleList(
            [
                SpikingTransformerBlock(
                    d_model,
                    n_head,
                    d_ff,
                    threshold,
                    decay,
                    amplitude,
                    use_surrogate,
                    ternary,
                    fast_weight_fc2=fast_weight_fc2,
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        # LM Head: static (kalıcı) + fast (geçici) weight
        self.lm_head_static = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer("lm_head_fast", torch.zeros(vocab_size, d_model))
        self.lm_head_bias = nn.Parameter(torch.zeros(vocab_size))
        # Consolidation tracking
        self.consolidation_count = 0
        self.update_counter = 0

    def reset(self, batch_size, device):
        self.input_lif.reset(batch_size, self.d_model, device)
        for block in self.blocks:
            block.reset(batch_size, device)

    def get_lm_head_weight(self):
        """Static + fast weight birleşimi."""
        return self.lm_head_static.weight + self.lm_head_fast

    def forward(self, idx, targets=None):
        B, T = idx.size()
        device = idx.device

        self.reset(B, device)

        # Token-by-token processing (RNN tarzı)
        all_logits = []
        all_hidden = []
        W = self.get_lm_head_weight()
        for t in range(T):
            tok_emb = self.wte(idx[:, t])  # (B, d_model)
            pos_emb = self.wpe(torch.tensor([t], device=device))  # (1, d_model)
            x = tok_emb + pos_emb  # (B, d_model)

            # Input → spike
            x = self.input_lif(x)  # (B, d_model) spike

            for block in self.blocks:
                x = block(x)  # (B, d_model)

            x = self.ln_f(x)
            all_hidden.append(x)
            logits_t = F.linear(x, W, self.lm_head_bias)  # (B, vocab)
            all_logits.append(logits_t)

        logits = torch.stack(all_logits, dim=1)  # (B, T, vocab)
        self._last_hidden = torch.stack(all_hidden, dim=1)  # (B, T, d_model)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def consolidate(self, alpha=0.05, threshold=0.5):
        """
        Layer-wise Episodic Memory Leak: Tüm fast weight'leri static'e sızdır.

        Returns:
            dict: Consolidation raporu (per-layer)
        """
        reports = {}
        total_consolidated = 0

        # LM Head consolidation
        fast_norm = self.lm_head_fast.norm().item()
        importance = fast_norm / max(self.lm_head_static.weight.norm().item(), 1e-8)

        if importance > threshold * 0.1:
            self.lm_head_static.weight.data = (
                1 - alpha
            ) * self.lm_head_static.weight.data + alpha * self.lm_head_fast
            self.lm_head_fast.zero_()
            self.consolidation_count += 1
            total_consolidated += 1
            reports["lm_head"] = {
                "status": "consolidated",
                "fast_norm": fast_norm,
                "importance": importance,
            }

        # Layer-wise FFN fc2 consolidation
        for li, block in enumerate(self.blocks):
            fc2 = block.ffn.fc2
            if fc2.fast_weight:
                layer_report = fc2.consolidate(alpha=alpha)
                if layer_report["status"] == "consolidated":
                    total_consolidated += 1
                reports[f"block_{li}_fc2"] = layer_report

        reports["total_consolidated"] = total_consolidated
        reports["consolidation_count"] = self.consolidation_count

        return reports

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def spike_rate(self):
        """Ortalama spike oranını hesapla (enerji verimliliği göstergesi)."""
        total_spikes = 0
        total_neurons = 0
        for name, module in self.named_modules():
            if isinstance(module, LIFNeuron) and module.v is not None:
                total_spikes += (module.v > module.threshold).float().sum().item()
                total_neurons += module.v.numel()
        return total_spikes / max(total_neurons, 1)

    def get_last_hidden(self):
        """Son forward pass'teki hidden state'i döndür (B, T, d_model)."""
        return getattr(self, "_last_hidden", None)

    def get_internal_plastic_layers(self):
        """
        LM head dışındaki tüm plastik katmanları döndür.
        STDP update'leri bu katmanlara uygulanır.
        """
        layers = []
        for block in self.blocks:
            layers.append(block.ffn.fc2)
            if block.ffn.fc2.fast_weight:
                layers.append(block.ffn.fc2)
        return list(set(layers))
