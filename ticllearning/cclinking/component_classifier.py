import torch
from torch import nn

from .hasse_message_passing import WeightedHasseMP

class CellClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()

        self.encoder = nn.Linear(in_channels, hidden_channels, dtype=torch.float64)

        self.layers = nn.ModuleList(
            WeightedHasseMP(hidden_channels)
            for _ in range(num_layers)
        )

        self.decoder = nn.Linear(hidden_channels, out_channels, dtype=torch.float64)


    def add_scaler(self, node_scaler):
        self.register_buffer("node_scaler", node_scaler)


    def forward(self, x, hasse_laplacian):
        x = x / self.node_scaler
        x = self.encoder(x)


        for layer in self.layers:
            x = x + layer.forward(x, hasse_laplacian)

        x = self.decoder(x)
        return x

