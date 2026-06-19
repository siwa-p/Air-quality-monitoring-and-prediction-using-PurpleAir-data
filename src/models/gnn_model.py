import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class STGNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        gru_layers: int = 2,
        horizon: int = 1,
    ):
        super().__init__()
        self.gat = GATConv(
            in_features, hidden, heads=heads, dropout=dropout,
            concat=True, edge_dim=1,
        )
        self.layer_norm = nn.LayerNorm(hidden * heads)
        self.gru = nn.GRU(
            input_size=hidden * heads,
            hidden_size=hidden,
            num_layers=gru_layers,
            batch_first=False,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ):
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)

        B, T, N, F_in = x.shape
        E = edge_index.shape[1]

        offsets = torch.arange(B, device=edge_index.device).view(B, 1, 1) * N
        batch_edge = (edge_index.unsqueeze(0).expand(B, -1, -1) + offsets).reshape(2, B * E)

        edge_attr = edge_weight.unsqueeze(-1).repeat(B, 1) if edge_weight is not None else None

        gat_outs = [
            self.gat(x[:, t].reshape(B * N, F_in), batch_edge, edge_attr=edge_attr)
            for t in range(T)
        ]

        # LayerNorm before dropout stabilises scale across stacked GRU layers
        seq = self.dropout(self.layer_norm(torch.stack(gat_outs, dim=0)))  # [T, B*N, heads*hidden]
        gru_out, _ = self.gru(seq)

        last = gru_out[-1].view(B, N, -1)
        out = self.head(last)               # [B, N, H]
        return out.squeeze(0) if squeeze else out
