import torch
from torch import nn
import torch.nn.functional as F

from topomodelx.base.message_passing import MessagePassing

class WeightedHasseMP(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr_func="sum", att=False)

        self.message_nn = nn.Sequential(
            nn.Linear(channels, channels, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(channels, channels, dtype=torch.float64),
        )

        self.att_nn = nn.Sequential(
            nn.Linear(2 * channels + 1, channels, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(channels, 1, dtype=torch.float64),
        )

        self.update = nn.Sequential(
            nn.Linear(2 * channels, channels, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(channels, channels, dtype=torch.float64),
        )

    def message(self, x_source, x_target=None):
        return self.message_nn(x_source)

    def forward(self, x, A):
        A = A.coalesce()

        edge_index = A.indices()
        edge_weight = A.values().to(dtype=x.dtype)

        src = edge_index[1]
        dst = edge_index[0]
        mask = src != dst

        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

        src = src[mask]
        dst = dst[mask]

        h_src = x[src]
        h_dst = x[dst]

        att_input = torch.cat(
            [h_dst, h_src, edge_weight[:, None]],
            dim=-1,
        )

        alpha = torch.sigmoid(self.att_nn(att_input)).squeeze(-1)
        effective_weight = edge_weight * alpha
        A_eff = torch.sparse_coo_tensor(
            edge_index,
            effective_weight,
            size=A.shape,
            device=x.device,
            dtype=x.dtype,
        ).coalesce()

        msg = super().forward(
            x_source=x,
            neighborhood=A_eff,
            x_target=x,
        )

        return self.update(torch.cat([x, msg], dim=-1))
