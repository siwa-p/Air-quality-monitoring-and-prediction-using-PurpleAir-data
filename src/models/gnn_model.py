import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class STGNN(nn.Module):
    """
    Spatio-Temporal GNN: GATConv per timestep + GRU across timesteps.

    Input:  x        — [T, N, F]  (timesteps, nodes, features)
            edge_index [2, E]
            edge_weight [E]  (optional)
    Output: [N, 1]   next-step pm2_5 per sensor node
    """

    def __init__(self, in_features: int, hidden: int = 64, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat = GATConv(in_features, hidden, heads=heads, dropout=dropout, concat=True)
        self.gru = nn.GRU(input_size=hidden * heads, hidden_size=hidden, batch_first=False)
        self.head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None):
        # Accept [T, N, F] (single) or [B, T, N, F] (batched)
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)

        B, T, N, F_in = x.shape
        E = edge_index.shape[1]

        # Shift edge indices by i*N for each graph in the batch
        offsets = torch.arange(B, device=edge_index.device).view(B, 1, 1) * N  # [B,1,1]
        batch_edge = (edge_index.unsqueeze(0).expand(B, -1, -1) + offsets).reshape(2, B * E)

        gat_outs = []
        for t in range(T):
            node_feat = x[:, t].reshape(B * N, F_in)     # [B*N, F]
            out = self.gat(node_feat, batch_edge)          # [B*N, heads*hidden]
            gat_outs.append(out)

        # [T, B*N, heads*hidden] → GRU → [T, B*N, hidden]
        seq = self.dropout(torch.stack(gat_outs, dim=0))
        gru_out, _ = self.gru(seq)

        last = gru_out[-1].view(B, N, -1)  # [B, N, hidden]
        out = self.head(last)               # [B, N, 1]
        return out.squeeze(0) if squeeze else out
