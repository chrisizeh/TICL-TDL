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

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        if attention:
            self.connection_attention = ReweightConnection(hidden_channels, num_ranks)
        else:
            self.connection_attention = Placeholder()

        self.layers = nn.ModuleList(
            HasseMP(hidden_channels, num_ranks)
            for _ in range(num_layers)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def add_scaler(self, node_scaler):
        self.register_buffer("node_scaler", node_scaler)


    def forward(self, x, L, ranks, rank3_cells):
        x = (x / self.node_scaler).float()
        x = self.encoder(x)

        L = self.connection_attention(x, L, ranks, rank3_cells)
        x = x[:-rank3_cells]
        for layer in self.layers:
            x = x + layer.forward(x, L, ranks[:-rank3_cells])

        x = self.decoder(x)
        return x

