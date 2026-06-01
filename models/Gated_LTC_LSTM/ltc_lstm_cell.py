"""
ltc_lstm_cell.py
================
Gated-LTC-LSTM Cell — kết hợp:
  1. Gated Input Layer  : Gate Sigmoid + Gate tanh (theo Gated-LNN paper, ref/)
  2. LTC ODE dynamics   : Liquid Time-Constant, Euler solver (Hasani et al.)
  3. LSTM Gating        : Forget / Input / Output gates kiểm soát cell state

Tham khảo:
  - Hasani et al., "Liquid Time-constant Networks", AAAI 2021
  - Gated-LNN paper: "Gated Liquid Neural Networks for Accurate Water Quality
    Index Prediction and Classification" (ref/Gated-LNN_...)
  - raminmh/liquid_time_constant_networks (GitHub)

Phương trình cốt lõi:
  Gated input:
      g_t  = sigmoid(W_g · x_t + b_g)
      z_t  = tanh(W_z · x_t + b_z)
      u_t  = g_t ⊙ z_t

  LSTM gates (dùng h_{t-1} và u_t):
      f_t  = sigmoid(W_f · [h_{t-1}, u_t] + b_f)   # forget
      i_t  = sigmoid(W_i · [h_{t-1}, u_t] + b_i)   # input
      o_t  = sigmoid(W_o · [h_{t-1}, u_t] + b_o)   # output

  LTC ODE candidate (thay cell candidate trong LSTM chuẩn):
      τ_t  = τ_min + softplus(W_τ · [h_{t-1}, u_t] + b_τ)   # liquid time constant
      A_t  = sigmoid(W_A · [h_{t-1}, u_t] + b_A)             # amplitude
      ĉ_t  = A_t ⊙ sigmoid(W_c · c_{t-1} + W_u · u_t + b_c) # target state
      -- Euler ODE step (n_steps steps) --
      Δτ   = 1.0 / (τ_t · n_steps)
      c̃_t  = c_{t-1} + Δτ · (-c_{t-1} + ĉ_t)  × n_steps

  Cell & hidden:
      c_t  = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
      h_t  = o_t ⊙ tanh(c_t)
"""

import torch
import torch.nn as nn
import math


class GatedLTCLSTMCell(nn.Module):
    """
    Một time-step của Gated-LTC-LSTM.

    Args:
        input_size  : số features đầu vào (58 sau preprocessing)
        hidden_size : kích thước hidden state
        tau_min     : time constant tối thiểu (>0), default=1.0
        n_ode_steps : số bước Euler tích phân ODE, default=6
    """

    def __init__(self, input_size: int, hidden_size: int,
                 tau_min: float = 1.0, n_ode_steps: int = 6):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.tau_min     = tau_min
        self.n_ode_steps = n_ode_steps

        # ── Gated Input Layer (từ Gated-LNN paper) ──────────────────────────
        # Chiếu input xuống hidden_size trước khi vào liquid layer
        self.W_g = nn.Linear(input_size, hidden_size)   # gate sigmoid
        self.W_z = nn.Linear(input_size, hidden_size)   # gate tanh

        # ── LSTM Gates (dùng concatenate [h, u]) ────────────────────────────
        combined = hidden_size + hidden_size  # h_{t-1} concat u_t
        self.W_f = nn.Linear(combined, hidden_size)  # forget gate
        self.W_i = nn.Linear(combined, hidden_size)  # input  gate
        self.W_o = nn.Linear(combined, hidden_size)  # output gate

        # ── LTC ODE Parameters ──────────────────────────────────────────────
        self.W_tau = nn.Linear(combined, hidden_size)  # liquid time constant
        self.W_A   = nn.Linear(combined, hidden_size)  # amplitude
        self.W_c   = nn.Linear(hidden_size, hidden_size)  # c_{t-1} → ĉ
        self.W_u   = nn.Linear(hidden_size, hidden_size)  # u_t     → ĉ
        self.b_c   = nn.Parameter(torch.zeros(hidden_size))

        self._reset_parameters()

    def _reset_parameters(self):
        """Khởi tạo trọng số theo cách phù hợp với LSTM/ODE."""
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        # Bias forget gate → 1 để tránh vanishing gradient ban đầu
        nn.init.ones_(self.W_f.bias)

    def forward(self, x_t, h_prev, c_prev):
        """
        Args:
            x_t    : (batch, input_size)
            h_prev : (batch, hidden_size)
            c_prev : (batch, hidden_size)
        Returns:
            h_t, c_t : (batch, hidden_size) mỗi cái
        """
        # ── 1. Gated Input ───────────────────────────────────────────────────
        g = torch.sigmoid(self.W_g(x_t))       # (batch, hidden)
        z = torch.tanh(self.W_z(x_t))           # (batch, hidden)
        u_t = g * z                              # (batch, hidden)

        # ── 2. Concat [h_{t-1}, u_t] cho các gates ───────────────────────────
        hu = torch.cat([h_prev, u_t], dim=-1)   # (batch, 2*hidden)

        # ── 3. LSTM Gates ────────────────────────────────────────────────────
        f_t = torch.sigmoid(self.W_f(hu))       # forget gate
        i_t = torch.sigmoid(self.W_i(hu))       # input  gate
        o_t = torch.sigmoid(self.W_o(hu))       # output gate

        # ── 4. LTC ODE: tính cell candidate c̃ ───────────────────────────────
        # 4a. Liquid time constant τ (>= tau_min, đảm bảo stability)
        tau = self.tau_min + torch.nn.functional.softplus(self.W_tau(hu))

        # 4b. Amplitude A ∈ (0,1)
        A = torch.sigmoid(self.W_A(hu))

        # 4c. ODE target state ĉ = A * σ(W_c·c + W_u·u + b)
        c_hat = A * torch.sigmoid(
            self.W_c(c_prev) + self.W_u(u_t) + self.b_c
        )

        # 4d. Euler integration: dc/dt = (-c + c_hat) / τ
        c_tilde = c_prev
        dt = 1.0 / self.n_ode_steps
        for _ in range(self.n_ode_steps):
            dcdt    = (-c_tilde + c_hat) / tau
            c_tilde = c_tilde + dt * dcdt       # Euler step

        # ── 5. Cell & Hidden State ────────────────────────────────────────────
        c_t = f_t * c_prev + i_t * c_tilde      # LSTM cell update
        h_t = o_t * torch.tanh(c_t)             # hidden output

        return h_t, c_t

    def init_hidden(self, batch_size, device):
        """Khởi tạo trạng thái ẩn về 0."""
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, c
