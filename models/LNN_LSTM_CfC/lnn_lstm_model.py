"""
lnn_lstm_model.py  (v2 — CfC-based)
"""

import torch
import torch.nn as nn
from model.lnn_lstm_cell import LNNLSTMCell


class LNNLSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 n_layers: int = 2, dropout: float = 0.2,
                 delta_t: float = 1.0):
        super().__init__()
        self.n_layers    = n_layers
        self.hidden_size = hidden_size

        # Layer 0 nhận input_size, các layer sau nhận hidden_size
        self.cells = nn.ModuleList([
            LNNLSTMCell(input_size if i == 0 else hidden_size,
                        hidden_size, delta_t)
            for i in range(n_layers)
        ])

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        device = x.device

        h_list, c_list = zip(*[cell.init_hidden(batch, device) for cell in self.cells])
        h_list, c_list = list(h_list), list(c_list)

        for t in range(seq_len):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h_new, c_new = cell(inp, h_list[i], c_list[i])
                inp = self.dropout(h_new) if i < self.n_layers - 1 else h_new
                h_list[i], c_list[i] = h_new, c_new

        return self.fc_out(self.layer_norm(h_list[-1]))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
