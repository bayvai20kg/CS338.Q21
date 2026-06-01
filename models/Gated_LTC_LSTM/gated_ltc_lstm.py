"""
gated_ltc_lstm.py
=================
Full model: Stacked Gated-LTC-LSTM cho bài toán dự báo chuỗi thời gian.

Architecture:
    Input  : (batch, seq_len, input_size)
    Layer 1: GatedLTCLSTMCell  → hidden_1
    Layer 2: GatedLTCLSTMCell  → hidden_2  (stacked, dùng h từ layer 1)
    Dropout → Linear(hidden → 1) → output

Tham khảo architecture tổng thể từ:
    - raminmh/liquid_time_constant_networks (GitHub)
    - Gated-LNN paper (ref/Gated-LNN_...)
"""

import torch
import torch.nn as nn
from model.ltc_lstm_cell import GatedLTCLSTMCell


class GatedLTCLSTM(nn.Module):
    """
    Stacked Gated-LTC-LSTM model cho time-series regression.

    Args:
        input_size  : số features (58)
        hidden_size : hidden dimension mỗi layer (default: 64)
        n_layers    : số lớp stacked (default: 2)
        dropout     : dropout rate giữa các layer (default: 0.2)
        tau_min     : minimum time constant cho LTC (default: 1.0)
        n_ode_steps : số Euler steps cho ODE (default: 6)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 n_layers: int = 2, dropout: float = 0.2,
                 tau_min: float = 1.0, n_ode_steps: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        # Stacked LTC-LSTM cells
        # Layer đầu: input_size → hidden_size
        # Các layer sau: hidden_size → hidden_size
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            in_sz = input_size if i == 0 else hidden_size
            self.cells.append(
                GatedLTCLSTMCell(in_sz, hidden_size, tau_min, n_ode_steps)
            )

        self.dropout   = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)  # ổn định training

        # Output head: hidden → PAC concentration (scalar)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),                          # Swish activation — smooth
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x, return_sequence: bool = False):
        """
        Args:
            x               : (batch, seq_len, input_size)
            return_sequence : nếu True, trả về output cho mỗi timestep
        Returns:
            out : (batch, 1) nếu return_sequence=False
                  (batch, seq_len, 1) nếu return_sequence=True
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # Khởi tạo hidden/cell states cho tất cả layers
        h_list = []
        c_list = []
        for cell in self.cells:
            h, c = cell.init_hidden(batch_size, device)
            h_list.append(h)
            c_list.append(c)

        outputs = []

        # Unroll theo thời gian
        for t in range(seq_len):
            x_t = x[:, t, :]          # (batch, input_size)

            for layer_idx, cell in enumerate(self.cells):
                h_new, c_new = cell(x_t, h_list[layer_idx], c_list[layer_idx])

                # Dropout giữa các layers (không áp dụng ở layer cuối)
                if layer_idx < self.n_layers - 1:
                    x_t = self.dropout(h_new)
                else:
                    x_t = h_new

                h_list[layer_idx] = h_new
                c_list[layer_idx] = c_new

            # Layer norm trên hidden state cuối
            h_out = self.layer_norm(h_list[-1])
            outputs.append(h_out)

        if return_sequence:
            # Stack: (batch, seq_len, hidden)
            out_seq = torch.stack(outputs, dim=1)
            return self.fc_out(out_seq)               # (batch, seq_len, 1)
        else:
            # Chỉ lấy timestep cuối
            return self.fc_out(outputs[-1])            # (batch, 1)

    def count_parameters(self):
        """Đếm tổng số trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
