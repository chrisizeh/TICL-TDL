import torch
from torch import nn

from .hasse_message_passing import HasseMP

class CellClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_ranks):
        super().__init__()

        self.encoder = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList(
            HasseMP(hidden_channels, num_ranks)
            for _ in range(num_layers)
        )

        self.decoder = nn.Linear(hidden_channels, out_channels)


    def add_scaler(self, node_scaler):
        self.register_buffer("node_scaler", node_scaler)


    def forward(self, x, L, ranks):
        L = L.to_dense()
        x = x / self.node_scaler
        x = self.encoder(x)

        for layer in self.layers:
            x = x + layer.forward(x, L, ranks)

        x = self.decoder(x)
        return x

