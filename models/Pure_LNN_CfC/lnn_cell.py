"""
lnn_cell.py
===========
Pure LNN Cell — Closed-form Continuous-time (CfC)

Đây là công thức LNN GỐC từ hai paper chính trong ref/:
  - "Liquid Time-constant Networks" (Hasani et al., AAAI 2021)
  - "Closed-form continuous-time neural networks" (Hasani et al., NeurIPS 2022)

Ý tưởng cốt lõi:
  LTC ODE:  dx/dt = [-x + f(x,I)·(A - x)] / τ
  
  CfC giải ODE này dưới dạng closed-form (không cần Euler), 
  cho ra nghiệm giải tích chính xác:

    h(t) = σ(-f·t + g) · src  +  (1 - σ(-f·t + g)) · attractor
           ↑ gate(t)                 ↑ 1 - gate(t)

  Trong đó:
    f       = softplus(W_f · [x, h_prev])   ← time constant (luôn dương)
    g       = W_g · [x, h_prev]             ← shift / bias
    src     = tanh(W_h  · [x, h_prev])      ← "what I respond to now"
    attract = tanh(W_hf · [x, h_prev])      ← "where I converge to"
    t       = delta_t (bước thời gian, =1 cho daily data)

Tại sao CfC beat LSTM:
  - Không có discretization error (Euler dễ bị lỗi tích lũy)
  - Multi-scale dynamics: f lớn → gate≈0 → bám attractor nhanh (turbidity đột biến)
                          f nhỏ → gate≈0.5 → hội tụ chậm (mùa vụ)
  - Gradient luôn sạch (không vanish qua nhiều bước Euler)
  - Cùng số params với LSTM nhưng expressiveness cao hơn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LNNCell(nn.Module):
    """
    Closed-form LNN cell (pure — không có LSTM gates).

    Args:
        input_size  : số features đầu vào
        hidden_size : kích thước hidden state
        delta_t     : bước thời gian (1.0 cho daily data)
    """

    def __init__(self, input_size: int, hidden_size: int, delta_t: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.delta_t     = delta_t

        combined = input_size + hidden_size

        # 4 networks của CfC — cùng structure với 4 gates LSTM (so sánh công bằng)
        self.W_f   = nn.Linear(combined, hidden_size)  # time constant
        self.W_g   = nn.Linear(combined, hidden_size)  # shift/bias
        self.W_h   = nn.Linear(combined, hidden_size)  # source state
        self.W_hf  = nn.Linear(combined, hidden_size)  # attractor state

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x      : (batch, input_size)
            h_prev : (batch, hidden_size)
        Returns:
            h_new  : (batch, hidden_size)
        """
        xh = torch.cat([x, h_prev], dim=-1)         # (batch, input+hidden)

        # ── Liquid dynamics ──────────────────────────────────────────────────
        f       = F.softplus(self.W_f(xh))           # time constant (>0)
        g       = self.W_g(xh)                        # shift/bias (free)
        src     = torch.tanh(self.W_h(xh))            # source state
        attract = torch.tanh(self.W_hf(xh))           # attractor state

        # ── Closed-form solution to LTC ODE ──────────────────────────────────
        # gate = σ(-f·Δt + g) — interpolates between src and attractor
        gate  = torch.sigmoid(-f * self.delta_t + g)
        h_new = gate * src + (1.0 - gate) * attract

        return h_new

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)
