
import torch
from torch import nn

from topomodelx.base.message_passing import MessagePassing
from ticllearning.cclinking.hasse_message_passing import HasseMP
from ticllearning.cclinking.k_missing_rewirer import KMissingRewirer
from ticllearning.cclinking.connection_attention import ReweightConnection

class CellClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_ranks, num_added_edges):
        super().__init__()

        self.encoder = nn.Linear(in_channels, hidden_channels)

        self.rewirer = KMissingRewirer(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_ranks=num_ranks,
            k=num_added_edges,
            max_added_weight=1.0,
            symmetric=True,
        )

        self.layers = nn.ModuleList(
            HasseMP(hidden_channels, num_ranks)
            for _ in range(num_layers)
        )

        self.decoder = nn.Linear(hidden_channels, out_channels)


    def add_scaler(self, node_scaler):
        self.register_buffer("node_scaler", node_scaler)


    def forward(self, x, L, ranks):
        x = x / self.node_scaler
        x = self.encoder(x)

        A = self.rewirer(x, L, ranks)
        L = L + A
        for layer in self.layers:
            x = x + layer.forward(x, L, ranks)

        x = self.decoder(x)
        return x

