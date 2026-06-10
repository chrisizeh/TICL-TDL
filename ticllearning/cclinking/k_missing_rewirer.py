import torch
from torch import nn
import torch.nn.functional as F


class KMissingRewirer(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_ranks,
        rank_emb_dim=1,
        k=3,
        max_added_weight=1.0,
        symmetric=True,
    ):
        super().__init__()

        self.k = k
        self.max_added_weight = max_added_weight
        self.symmetric = symmetric

        #self.rank_emb = nn.Embedding(num_ranks, rank_emb_dim)

        pair_channels = (
            4 * in_channels       # x_i, x_j, |x_i-x_j|, x_i*x_j
            + 2 * in_channels     # neighbourhood summaries
            + 2 * rank_emb_dim    # rank_i, rank_j
            + 2                  # degree_i, degree_j
        )

        self.score_nn = nn.Sequential(
            nn.Linear(pair_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )


    def forward(self, x, L, ranks):
        n = x.shape[0]
        L = L.to_dense()

        degree = torch.diag(L)
        neigh = L @ x

        xi = x[:, None].expand(n, n, -1)
        xj = x[None, :].expand(n, n, -1)

        ni = neigh[:, None].expand(n, n, -1)
        nj = neigh[None, :].expand(n, n, -1)

        ri = ranks[:, None].expand_as(L).unsqueeze(-1)
        rj = ranks[None, :].expand_as(L).unsqueeze(-1)

        di = degree[:, None].expand_as(L).unsqueeze(-1)
        dj = degree[None, :].expand_as(L).unsqueeze(-1)

        pair_features = torch.cat(
            [
                xi,
                xj,
                torch.abs(xi - xj),
                xi * xj,
                ni,
                nj,
                ri,
                rj,
                di,
                dj,
            ],
            dim=-1,
        )
        scores = self.score_nn(pair_features[:, None]).squeeze(-1).squeeze(1)
        existing = L > 0
        eye = torch.eye(n, device=x.device, dtype=torch.bool)

        missing_mask = (~existing) & (~eye)
        scores = scores.masked_fill(~missing_mask, -torch.inf)

        k = min(self.k, n - 1)
        top_scores, top_src = torch.topk(scores, k=k, dim=1)

        dst = torch.arange(n, device=x.device).repeat_interleave(k)
        src = top_src.reshape(-1)
        selected_scores = top_scores.reshape(-1)

        valid = torch.isfinite(selected_scores)

        dst = dst[valid]
        src = src[valid]
        selected_scores = selected_scores[valid]

        weights = self.max_added_weight * torch.sigmoid(selected_scores)

        A_added = x.new_zeros(n, n)
        A_added[dst, src] = weights

        if self.symmetric:
            A_added = torch.maximum(A_added, A_added.T)

        edge_index_added = torch.nonzero(A_added, as_tuple=False).T
        edge_weight_added = A_added[edge_index_added[0], edge_index_added[1]]

        return edge_index_added, edge_weight_added, A_added
