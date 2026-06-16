import os
import os.path as osp
import argparse
import torch
import pandas as pd
from torch_geometric.loader.dataloader import DataLoader

from scripts.CONFIG import CONFIG
from ticllearning.cclinking.component_classifier import CellClassifier
from ticllearning.datasets.ccnn.dataset import CCDataset


def to_dense(x):
    if x.is_sparse:
        return x.coalesce().to_dense()
    return x


def remove_diag(M):
    M = M.clone()
    eye = torch.eye(M.shape[0], device=M.device, dtype=torch.bool)
    M[eye] = 0
    return M


def top_entries(M, ranks, k=30, mask=None):
    M = M.detach()
    if mask is not None:
        scores = M.masked_fill(~mask, -float("inf"))
    else:
        scores = M.clone()

    scores = remove_diag(scores)

    vals, idx = torch.topk(scores.flatten(), k=min(k, scores.numel()))
    rows = idx // M.shape[1]
    cols = idx % M.shape[1]

    out = []
    for val, i, j in zip(vals, rows, cols):
        if torch.isinf(val):
            continue
        out.append({
            "src": int(j.item()),
            "dst": int(i.item()),
            "value": float(val.item()),
            "rank_src": int(ranks[j].item()),
            "rank_dst": int(ranks[i].item()),
        })
    return pd.DataFrame(out)


def rank_pair_stats(M, A, ranks, max_rank):
    rows = []
    eye = torch.eye(M.shape[0], device=M.device, dtype=torch.bool)

    for ri in range(max_rank + 1):
        for rj in range(max_rank + 1):
            mask = (ranks[:, None] == ri) & (ranks[None, :] == rj) & ~eye
            if not mask.any():
                continue

            vals = M[mask]
            old = A[mask] > 0
            new = A[mask] == 0

            row = {
                "rank_dst": ri,
                "rank_src": rj,
                "count": int(mask.sum().item()),
                "mean": float(vals.mean().item()),
                "max": float(vals.max().item()),
                "above_001": int((vals > 0.01).sum().item()),
                "above_005": int((vals > 0.05).sum().item()),
                "above_010": int((vals > 0.10).sum().item()),
            }

            if old.any():
                row["old_mean"] = float(vals[old].mean().item())
                row["old_max"] = float(vals[old].max().item())
            else:
                row["old_mean"] = None
                row["old_max"] = None

            if new.any():
                row["new_mean"] = float(vals[new].mean().item())
                row["new_max"] = float(vals[new].max().item())
                row["new_above_010"] = int((vals[new] > 0.10).sum().item())
            else:
                row["new_mean"] = None
                row["new_max"] = None
                row["new_above_010"] = 0

            rows.append(row)

    return pd.DataFrame(rows)


def global_stats(A, alpha, C, C_adj):
    eye = torch.eye(A.shape[0], device=A.device, dtype=torch.bool)

    old = (A > 0) & ~eye
    new = (A == 0) & ~eye

    rows = []

    for name, M in {
        "alpha": alpha,
        "C": C,
        "C_adj": C_adj,
    }.items():
        M = remove_diag(M)

        row = {
            "matrix": name,
            "mean": float(M[~eye].mean().item()),
            "max": float(M[~eye].max().item()),
            "old_mean": float(M[old].mean().item()) if old.any() else None,
            "old_max": float(M[old].max().item()) if old.any() else None,
            "new_mean": float(M[new].mean().item()) if new.any() else None,
            "new_max": float(M[new].max().item()) if new.any() else None,
        }

        for th in [0.01, 0.05, 0.10, 0.20, 0.50]:
            row[f"old_above_{th}"] = int((M[old] > th).sum().item()) if old.any() else 0
            row[f"new_above_{th}"] = int((M[new] > th).sum().item()) if new.any() else 0

        rows.append(row)

    return pd.DataFrame(rows)


def analyse_sample(model, sample, out_dir, sample_idx, max_rank, top_k):
    model.eval()

    with torch.no_grad():
        sample = sample.to(CONFIG.device)
        _, debug = model(sample.x, sample.A, sample.ranks, sample.num_cells)

    A = to_dense(debug["A"]).detach().float()
    alpha = to_dense(debug["alpha"]).detach().float()
    C = to_dense(debug["C"]).detach().float()
    C_adj = to_dense(debug["C_adj"]).detach().float()
    ranks = sample.ranks.detach().long()

    n = A.shape[0]
    eye = torch.eye(n, device=A.device, dtype=torch.bool)
    old_mask = (A > 0) & ~eye
    new_mask = (A == 0) & ~eye

    sample_dir = osp.join(out_dir, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)

    global_stats(A, alpha, C, C_adj).to_csv(
        osp.join(sample_dir, "global_stats.csv"),
        index=False,
    )

    rank_pair_stats(C, A, ranks, max_rank).to_csv(
        osp.join(sample_dir, "rank_pair_stats_C.csv"),
        index=False,
    )

    rank_pair_stats(C_adj, A, ranks, max_rank).to_csv(
        osp.join(sample_dir, "rank_pair_stats_C_adj.csv"),
        index=False,
    )

    top_entries(C, ranks, k=top_k, mask=new_mask).to_csv(
        osp.join(sample_dir, "top_new_edges_C.csv"),
        index=False,
    )

    top_entries(C, ranks, k=top_k, mask=old_mask).to_csv(
        osp.join(sample_dir, "top_reweighted_old_edges_C.csv"),
        index=False,
    )

    top_entries(C_adj, ranks, k=top_k).to_csv(
        osp.join(sample_dir, "top_connectable_pairs_C_adj.csv"),
        index=False,
    )

    deg_A = A.sum(dim=1)
    deg_C = C.sum(dim=1)
    deg_df = pd.DataFrame({
        "cell": list(range(n)),
        "rank": ranks.cpu().tolist(),
        "deg_A": deg_A.cpu().tolist(),
        "deg_C": deg_C.cpu().tolist(),
        "delta_deg": (deg_C - deg_A).cpu().tolist(),
    })
    deg_df.to_csv(osp.join(sample_dir, "degree_comparison.csv"), index=False)

    print(f"\nSample {sample_idx}")
    print(global_stats(A, alpha, C, C_adj))
    print("\nStrongest new C edges")
    print(top_entries(C, ranks, k=10, mask=new_mask))


def main():
    experiment_name = "laplacian_build_cell_linking"
    data_info = "closeby_multi_0pu"
    num_samples = 5
    top_k = 50


    model = CellClassifier(
        in_channels=[6, 16, 16],
        hidden_channels=32,
        out_channels=1,
        num_layers=2,
        num_ranks=3,
        attention=True,
    ).to(CONFIG.device)

    date = "2026-06-16"
    extra_info = "epoch_3_dict"
    run_name = f"{date}_{experiment_name}_{data_info}"
    weights = torch.load(osp.join(CONFIG.model, run_name, f"{run_name}_{extra_info}.pt"), weights_only=True)
    model.load_state_dict(weights["model_state_dict"], strict=False)

    out_dir = osp.join(CONFIG.plots, f"analysis_{run_name}_{extra_info}")
    os.makedirs(out_dir, exist_ok=True)

    train_dataset = CCDataset(data_info, CONFIG, test=False)
    dataset = CCDataset(data_info, CONFIG, test=True)
    loader = DataLoader(dataset, shuffle=False, batch_size=1)
    model.add_scaler(train_dataset.node_scaler)

    for sample_idx, sample in enumerate(loader):
        analyse_sample(
            model=model,
            sample=sample,
            out_dir=out_dir,
            sample_idx=sample_idx,
            max_rank=3,
            top_k=top_k,
        )

    print(f"\nSaved analysis to {out_dir}")


if __name__ == "__main__":
    main()
