"""
lnn_lstm_cell.py  (v2 — CfC analytical solution)
=================================================
LNN+LSTM: Thay thế Euler ODE bằng CfC (Closed-form Continuous-time).

Vấn đề phiên bản cũ:
  - Euler ODE (6 steps) tích lũy lỗi xấp xỉ → gradient nhiễu → khó hội tụ

Giải pháp:
  - Dùng CfC (Closed-form) — nghiệm giải tích CHÍNH XÁC của LTC ODE
  - Thay thế LSTM cell candidate (tanh gate) bằng CfC liquid candidate
  - Giữ nguyên LSTM forget/input/output gates (không có gated input layer)

Công thức CfC cell candidate (Hasani et al., NeurIPS 2022):
  f     = softplus(W_f · [h, x])        ← liquid time constant (>0)
  g     = W_g · [h, x]                  ← bias/shift
  src   = tanh(W_s · [h, x])            ← "what I respond to NOW"
  att   = tanh(W_a · [h, x])            ← "where I converge TO"
  gate  = σ(-f·Δt + g)                  ← interpolation weight
  c̃    = gate·src + (1-gate)·att        ← liquid cell candidate

Rồi kết hợp LSTM:
  c_t = forget·c_{t-1} + input·c̃
  h_t = output·tanh(c_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LNNLSTMCell(nn.Module):
    """
    LNN (CfC) + LSTM cell — không Gated Input, không Euler.

    Args:
        input_size  : số features (58 cho layer đầu, hidden cho layer sau)
        hidden_size : hidden dimension
        delta_t     : timestep (1.0 cho daily data)
    """

    def __init__(self, input_size: int, hidden_size: int, delta_t: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.delta_t     = delta_t

        combined = input_size + hidden_size   # input + hidden, không cần projection

        # ── LSTM Gates (standard) ─────────────────────────────────────────────
        self.W_forget = nn.Linear(combined, hidden_size)  # forget gate
        self.W_input  = nn.Linear(combined, hidden_size)  # input  gate
        self.W_output = nn.Linear(combined, hidden_size)  # output gate

        # ── CfC Cell Candidate (LNN — thay thế tanh candidate của LSTM) ──────
        self.W_cf = nn.Linear(combined, hidden_size)  # time constant (f)
        self.W_cg = nn.Linear(combined, hidden_size)  # bias/shift    (g)
        self.W_cs = nn.Linear(combined, hidden_size)  # source state  (src)
        self.W_ca = nn.Linear(combined, hidden_size)  # attractor     (att)

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        # Forget bias = 1 → tránh vanishing gradient lúc đầu
        nn.init.ones_(self.W_forget.bias)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor,
                c_prev: torch.Tensor):
        """
        Args:
            x_t    : (batch, input_size)
            h_prev : (batch, hidden_size)
            c_prev : (batch, hidden_size)
        Returns:
            h_t, c_t : (batch, hidden_size)
        """
        # Concat x_t với h_prev — không qua gating hay projection đặc biệt
        xh = torch.cat([x_t, h_prev], dim=-1)   # (batch, input+hidden)

        # ── 1. LSTM Gates ─────────────────────────────────────────────────────
        f_gate = torch.sigmoid(self.W_forget(xh))
        i_gate = torch.sigmoid(self.W_input(xh))
        o_gate = torch.sigmoid(self.W_output(xh))

        # ── 2. CfC Liquid Cell Candidate ─────────────────────────────────────
        # 2a. Liquid time constant: luôn > 0 nhờ softplus
        f = F.softplus(self.W_cf(xh))

        # 2b. Bias/shift: cho phép âm/dương tự do
        g = self.W_cg(xh)

        # 2c. Source state: "what the cell responds to RIGHT NOW"
        src = torch.tanh(self.W_cs(xh))

        # 2d. Attractor: "where the cell converges TO over time"
        att = torch.tanh(self.W_ca(xh))

        # 2e. Interpolation gate (analytical LTC solution, no Euler)
        interp = torch.sigmoid(-f * self.delta_t + g)

        # 2f. CfC cell candidate = linear interpolation giữa src và attractor
        c_tilde = interp * src + (1.0 - interp) * att

        # ── 3. LSTM Cell & Hidden State ───────────────────────────────────────
        c_t = f_gate * c_prev + i_gate * c_tilde   # cell state
        h_t = o_gate * torch.tanh(c_t)             # hidden output

        return h_t, c_t

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, c
