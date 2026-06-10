
import torch
from torch import nn

class ReweightConnection(nn.Module):
    def __init__(self, channels, max_rank):
        super().__init__()
        self.max_rank = max_rank

        self.att_src = nn.Linear(channels, 1)
        self.att_dst = nn.Linear(channels, 1)
        self.att_w = nn.Sequential(
            nn.Linear(5, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )

    def forward(self, x, A, ranks):
        A = A.to_dense()
        A[A == 0] = 1e-8
        rank_i = ranks[None, :].expand_as(A)/self.max_rank
        rank_j = ranks[:, None].expand_as(A)/self.max_rank

        src = self.att_src(x).squeeze(1)[None, :].expand_as(A)
        dst = self.att_dst(x).squeeze(1)[:, None].expand_as(A)
        w = self.att_w(torch.cat([A[..., None], rank_i[..., None], rank_j[..., None], src[..., None], dst[..., None]], dim=-1)).squeeze(-1)

        alpha = src + dst + w 
        alpha = alpha.masked_fill(
            torch.eye(A.shape[0], device=A.device, dtype=torch.bool),
            0,
        )
        A = torch.softmax(alpha * A + A, dim=1)
        return A.to_sparse()
