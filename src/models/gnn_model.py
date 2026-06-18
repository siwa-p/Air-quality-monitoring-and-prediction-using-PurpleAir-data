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
        T, N, F = x.shape
        gat_outs = []
        for t in range(T):
            node_feat = x[t]                    # [N, F]
            out = self.gat(node_feat, edge_index)  # [N, heads*hidden]
            gat_outs.append(out)

        # Stack: [T, N, heads*hidden]
        seq = torch.stack(gat_outs, dim=0)
        seq = self.dropout(seq)

        # GRU: input [T, N, hidden*heads] → output [T, N, hidden]
        gru_out, _ = self.gru(seq)

        # Take last timestep: [N, hidden]
        last = gru_out[-1]
        return self.head(last)  # [N, 1]
