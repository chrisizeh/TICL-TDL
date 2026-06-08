import torch
from torch import nn
import torch.nn.functional as F

from topomodelx.base.message_passing import MessagePassing

class HasseMP(MessagePassing):
    def __init__(self, channels, max_rank):
        super().__init__(aggr_func="sum", att=False)
        self.max_rank = max_rank

        self.message_nn = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

        self.att_src = nn.Linear(channels, 1)
        self.att_dst = nn.Linear(channels, 1)
        self.att_w = nn.Sequential(
            nn.Linear(3, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )

        self.att_out = nn.Linear(channels, 1)

        self.update = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def message(self, x_source, x_target=None):
        return self.message_nn(x_source)

    def forward(self, x, A, ranks):
        rank_i = ranks[None, :].expand_as(A)/self.max_rank
        rank_j = ranks[:, None].expand_as(A)/self.max_rank

        src = self.att_src(x)
        dst = self.att_dst(x)
        w = self.att_w(torch.cat([A[..., None], rank_i[..., None], rank_j[..., None]], dim=-1)).squeeze(-1)

        alpha = src + dst + w 
        alpha = alpha.masked_fill(
            torch.eye(A.shape[0], device=A.device, dtype=torch.bool),
            -1e9,
        )
        alpha = torch.softmax(alpha, dim=1) * A
        msg = alpha @ self.message_nn(x)

        return self.update(torch.cat([x, msg], dim=-1))
