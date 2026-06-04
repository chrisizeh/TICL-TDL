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

        self.att_src = nn.Linear(channels, 1, dtype=torch.float64)
        self.att_dst = nn.Linear(channels, 1, dtype=torch.float64)
        self.att_w = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.att_out = nn.Linear(channels, 1, dtype=torch.float64)

        self.update = nn.Sequential(
            nn.Linear(2 * channels, channels, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(channels, channels, dtype=torch.float64),
        )

    def message(self, x_source, x_target=None):
        return self.message_nn(x_source)

    def forward(self, x, A):
        src = self.att_src(x)
        dst = self.att_dst(x)
        w = self.att_w(A[..., None]).squeeze(-1)

        alpha = src + dst + w 
        alpha = alpha.masked_fill(
            torch.eye(A.shape[0], device=A.device, dtype=torch.bool),
            -1e9,
        )
        alpha = torch.softmax(alpha, dim=1) * A.abs()
        msg = alpha @ self.message_nn(x)

        return self.update(torch.cat([x, msg], dim=-1))
