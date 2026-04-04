"""
VİCDAN_SPIKE — Reward-Modulated STDP (R-STDP)

STDP tek başına gürültü üretir. Reward sinyali ile yönlendirilmeli.

ΔW = reward × (A+ · pre · post - A- · pre · post)

Eğitim: Surrogate gradient (BPTT)
Inference: R-STDP (online adaptasyon)
"""

import torch
import torch.nn as nn


class RSTDP(nn.Module):
    """
    Reward-Modulated Spike-Timing-Dependent Plasticity.

    Her spike çifti için:
    - Doğru tahmin (reward > 0) → bağlantı güçlenir
    - Yanlış tahmin (reward < 0) → bağlantı zayıflar

    Time window: ±20ms (exponential decay)
    """

    def __init__(
        self,
        A_plus=0.01,
        A_minus=0.012,
        tau_plus=20.0,  # ms, LTP time constant
        tau_minus=20.0,  # ms, LTD time constant
        weight_min=-1.0,
        weight_max=1.0,
    ):
        super().__init__()
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.weight_min = weight_min
        self.weight_max = weight_max

        # Spike zamanları (her nöron için son spike zamanı)
        self.pre_spike_times = None
        self.post_spike_times = None

    def reset(self, n_pre, n_post, device):
        """Spike zamanlarını sıfırla."""
        self.pre_spike_times = torch.full((n_pre,), float("-inf"), device=device)
        self.post_spike_times = torch.full((n_post,), float("-inf"), device=device)

    def update(
        self,
        weight: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        reward: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        R-STDP weight güncellemesi.

        Args:
            weight: (out, in) — mevcut ağırlıklar
            pre_spikes: (in,) — pre nöron spike'ları (0 veya 1)
            post_spikes: (out,) — post nöron spike'ları (0 veya 1)
            reward: scalar — reward sinyali (-1 ile +1 arası)
            dt: timestep (ms)

        Returns:
            updated_weight: (out, in) — güncellenmiş ağırlıklar
        """
        if self.pre_spike_times is None:
            self.reset(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)

        # Spike zamanlarını güncelle
        current_time = (
            self.pre_spike_times.max() + dt
            if self.pre_spike_times.max() > float("-inf")
            else 0.0
        )

        # Pre spike'lar
        pre_active = pre_spikes > 0.5
        if pre_active.any():
            self.pre_spike_times[pre_active] = current_time

        # Post spike'lar
        post_active = post_spikes > 0.5
        if post_active.any():
            self.post_spike_times[post_active] = current_time

        # STDP update
        dw = torch.zeros_like(weight)

        for i in range(weight.shape[0]):  # post nöronlar
            if not post_active[i]:
                continue

            t_post = self.post_spike_times[i]

            for j in range(weight.shape[1]):  # pre nöronlar
                if not pre_active[j]:
                    continue

                t_pre = self.pre_spike_times[j]
                dt_spike = t_post - t_pre

                if dt_spike > 0:
                    # Pre önce → LTP (potentiation)
                    dw[i, j] = self.A_plus * torch.exp(
                        torch.tensor(-dt_spike / self.tau_plus)
                    )
                else:
                    # Post önce → LTD (depression)
                    dw[i, j] = -self.A_minus * torch.exp(
                        torch.tensor(dt_spike / self.tau_minus)
                    )

        # Reward ile modüle et
        dw = reward * dw

        # Weight update + clipping
        new_weight = torch.clamp(weight + dw, self.weight_min, self.weight_max)

        return new_weight


class SimpleRSTDP(nn.Module):
    """
    Basitleştirilmiş R-STDP (token-level, time window olmadan).

    Her token üretildiğinde çağrılır:
    ΔW = reward × (A+ · pre · post - A- · pre · (1 - post))

    Bu formül:
    - Aktif pre + aktif post → güçlenir (LTP)
    - Aktif pre + pasif post → zayıflar (LTD)
    - Reward > 0 → doğru bağlantılar güçlenir
    - Reward < 0 → yanlış bağlantılar zayıflar
    """

    def __init__(
        self,
        A_plus=0.01,
        A_minus=0.012,
        weight_min=-1.0,
        weight_max=1.0,
    ):
        super().__init__()
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.weight_min = weight_min
        self.weight_max = weight_max

    def update(
        self,
        weight: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        reward: float,
    ) -> torch.Tensor:
        """
        R-STDP weight güncellemesi.

        Args:
            weight: (out, in) — mevcut ağırlıklar
            pre_spikes: (in,) — pre nöron spike'ları (0 veya 1)
            post_spikes: (out,) — post nöron spike'ları (0 veya 1)
            reward: scalar — reward sinyali (0 ile 1 arası)

        Returns:
            updated_weight: (out, in)
        """
        # LTP: pre ve post birlikte aktif → güçlenir
        ltp = self.A_plus * torch.outer(post_spikes, pre_spikes)

        # LTD: pre aktif ama post pasif → zayıflar
        ltd = self.A_minus * torch.outer((1 - post_spikes), pre_spikes)

        # Combined update × reward
        dw = reward * (ltp - ltd)

        # Update + clip
        new_weight = torch.clamp(weight + dw, self.weight_min, self.weight_max)

        return new_weight


def shaped_reward(output_chars, target_suffix="lyn", max_reward=1.0):
    """
    Shaped reward: hedef suffix'e ne kadar yakınsa o kadar yüksek.

    Örn: target="lyn"
    "...lyn" → 1.0
    "...yn"  → 0.66
    "...n"   → 0.33
    "...x"   → 0.0
    """
    score = 0
    for i, (o, t) in enumerate(zip(reversed(output_chars), reversed(target_suffix))):
        if o == t:
            score += 1
        else:
            break
    return (score / len(target_suffix)) * max_reward
