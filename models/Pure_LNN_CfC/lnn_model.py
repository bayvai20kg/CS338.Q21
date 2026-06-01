"""
lnn_model.py
============
Stacked LNN model (pure CfC) cho time-series regression.

Architecture:
    Input : (batch, seq_len, input_size)
    Layer 1: LNNCell  →  h1_t
    Dropout + ResidualProj (nếu dim khác nhau)
    Layer 2: LNNCell  →  h2_t
    LayerNorm → Linear(hidden→1)
"""

import torch
import torch.nn as nn
from model.lnn_cell import LNNCell


class LNNModel(nn.Module):
    """
    Stacked CfC-LNN model.

    Args:
        input_size  : số features (58)
        hidden_size : hidden dimension (default: 64)
        n_layers    : số LNNCell stacked (default: 2)
        dropout     : dropout rate (default: 0.2)
        delta_t     : timestep cho CfC (default: 1.0 = 1 ngày)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 n_layers: int = 2, dropout: float = 0.2,
                 delta_t: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        # Stacked LNN cells
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            in_sz = input_size if i == 0 else hidden_size
            self.cells.append(LNNCell(in_sz, hidden_size, delta_t))

        # Input projection cho residual (layer 2 trở đi)
        # Nếu input_size != hidden_size cần project
        self.input_proj = nn.Linear(input_size, hidden_size) \
            if input_size != hidden_size else nn.Identity()

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Output head — giống LSTM baseline để so sánh công bằng
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, seq_len, input_size)
        Returns:
            out : (batch, 1)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # Init hidden states
        h_list = [cell.init_hidden(batch_size, device) for cell in self.cells]

        # Unroll theo thời gian
        for t in range(seq_len):
            x_t = x[:, t, :]   # (batch, input_size)

            for i, cell in enumerate(self.cells):
                h_new = cell(x_t, h_list[i])

                # Residual connection (skip): thêm projected input vào hidden
                if i == 0:
                    # Project input sang hidden space để cộng residual
                    res = self.input_proj(x[:, t, :])
                else:
                    res = h_list[i - 1]   # dùng hidden của layer trước

                h_new = h_new + 0.1 * res  # scaled residual (0.1 tránh dominant)

                # Dropout giữa layers
                if i < self.n_layers - 1:
                    x_t = self.dropout(h_new)
                else:
                    x_t = h_new

                h_list[i] = h_new

        # Lấy hidden state cuối cùng của layer cuối
        h_out = self.layer_norm(h_list[-1])
        return self.fc_out(h_out)   # (batch, 1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
