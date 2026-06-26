import torch
from torch import nn

from ticllearning.cclinking.hasse_message_passing import HasseMP
from ticllearning.cclinking.connection_attention import ReweightConnection


class Placeholder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, L, ranks):
        return L


class CellClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_ranks, attention=True):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.encoder_0 = nn.Sequential(
            nn.Linear(in_channels[0], hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.encoder_1 = nn.Sequential(
            nn.Linear(in_channels[1], hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.encoder_2 = nn.Sequential(
            nn.Linear(in_channels[2], hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        if attention:
            self.connection_attention = ReweightConnection(hidden_channels, num_ranks)
        else:
            self.connection_attention = Placeholder()

        self.layers = nn.ModuleList(HasseMP(hidden_channels, num_ranks) for _ in range(num_layers))

        self.cell_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

        self.rank2_merge_head = nn.Sequential(
            nn.Linear(4 * hidden_channels + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def add_scaler(self, node_scaler):
        self.register_buffer("node_scaler_0", node_scaler[0])
        self.register_buffer("node_scaler_1", node_scaler[1])
        self.register_buffer("node_scaler_2", node_scaler[2])

    def encode(self, x, L, ranks, num_cells):
        x[0] = (x[0] / self.node_scaler_0).float()
        x[0] = self.encoder_0(x[0])

        x[1] = (x[1] / self.node_scaler_1).float()
        x[1] = self.encoder_1(x[1])

        x[2] = (x[2] / self.node_scaler_2).float()
        x[2] = self.encoder_2(x[2])
        x.append(torch.zeros((num_cells[3], self.hidden_channels)))
        x = torch.cat(x, axis=0)

        L, debug = self.connection_attention(x, L, ranks, num_cells[3])
        x = x[: -num_cells[3]]
        for layer in self.layers:
            x = x + layer.forward(x, L, ranks[: -num_cells[3]])

        return x, L, debug

    def score_rank2_merges(self, x, edge_index, edge_weight=None):
        src, dst = edge_index

        xi = x[src]
        xj = x[dst]
        pair = torch.cat([xi, xj, torch.abs(xi - xj), xi * xj], dim=-1)

        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.shape[1])

        pair = torch.cat([pair, edge_weight[:, None]], dim=-1)

        return self.rank2_merge_head(pair).squeeze(-1)

    def forward(self, x, L, ranks, num_cells):
        x, L, debug = self.encode(x, L, ranks, num_cells)

        cell_logits = self.cell_head(x)
        edge_index = L.indices()
        edge_weight = L.values()
        row, col = edge_index
        keep = (ranks[row] == 2) & (ranks[col] == 2) & (row != col)

        edge_index = edge_index[:, keep]
        edge_weight = edge_weight[keep].abs()
        rank2_merge_logits = self.score_rank2_merges(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        return cell_logits, edge_index, rank2_merge_logits, debug
