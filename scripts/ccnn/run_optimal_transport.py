import os
import os.path as osp

import torch
from glob import glob
import copy

import matplotlib.pyplot as plt
import awkward as ak

from scripts.CONFIG import CONFIG
from ticllearning.cclinking.component_classifier import CellClassifier
from ticllearning.datasets.ccnn.dataset import CCDataset

from awkward_complex.classes.spectral import Spectral
from awkward_complex.datasets.cern.build import CERN

def validate_selection(connections, associations, num_rank2, cell_mask):
    M = torch.zeros((connections.shape[0], num_rank2))
    valid = connections >= 0
    rows = torch.arange(connections.shape[0])[valid]
    M[rows, connections[valid]] = 1
    y = (M @ associations[-num_rank2:]) == associations[:cell_mask.shape[0]][cell_mask]
    return y.sum()/y.shape[0]

def sinkhorn_from_cost(C, eps=0.05, n_iter=200):
    """
    C: cost matrix [n, m]
    returns transport plan P [n, m]
    """

    n, m = C.shape
    device = C.device
    dtype = C.dtype

    a = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    b = torch.full((m,), 1.0 / m, device=device, dtype=dtype)

    K = torch.exp(-C / eps)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)

    P = u[:, None] * K * v[None, :]
    return P

def build_model_data(cc):
    A = Spectral.graded_incidence_matrix(cc, weighted=True).to_dense()
    cell_mask = torch.ones(A.shape[0]).bool()
    cell_mask[:cc.num_nodes] = cc.get_connected_nodes_mask()
    A = A[cell_mask][:, cell_mask]

    x = cc.nodes
    x = x[cell_mask[:cc.num_nodes], :]
    num_cells = torch.Tensor([x.shape[0], cc._num_cells_at_rank(1), cc._num_cells_at_rank(2), cc._num_cells_at_rank(3)]).long()

    ranks = torch.cat([torch.zeros(cc.num_nodes), ak.to_torch(cc.cells.rank)]).to(cc.device)
    ranks = ranks[cell_mask]
    x = [x, ak.to_torch(cc.cells.features[cc.cells.rank == 1]).float().to(cc.device), ak.to_torch(cc.cells.features[cc.cells.rank == 2]).float().to(cc.device)]

    return x, A, ranks, num_cells


if __name__ == "__main__":
    data_info = "closeby_multi_0pu"
    model_name = "laplacian_build_cell_linking"
    model_date = "2026-06-16"
    extra_info = "epoch_29_dict"
    run_name = f"{model_date}_{model_name}_{data_info}"
    experiment_name = f"1_step_optimal_transport_{run_name}"
    os.makedirs(osp.join(CONFIG.plots, experiment_name), exist_ok=True)

    thresh = 0.6
    n_events = 10

    in_channels = [6, 16, 16]
    hidden_channels = 32
    num_classes = 1
    num_layer = 2

    model = CellClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layer,
        num_ranks=3,
        attention=True
    ).to(CONFIG.device)

    weights = torch.load(osp.join(CONFIG.model, run_name, f"{run_name}_{extra_info}.pt"), weights_only=True)
    model.load_state_dict(weights["model_state_dict"], strict=False)

    train_dataset = CCDataset(data_info, CONFIG, test=False)
    model.add_scaler(train_dataset.node_scaler)
    model.eval()

    input_folder = osp.join(CONFIG.histo, data_info)
    files = glob(f"{input_folder}/test/*.root")
    histo_data = CERN(files[0], CONFIG.device)

    base_vals = []
    adap_vals = []
    diffs = []
    with torch.no_grad():
        for sample in range(histo_data.n_events):
            cc = histo_data.build_cc(sample)
            cc = histo_data.add_skeleton_graph(cc, sample)
            base_cc, rank2_pos, rank2_children = histo_data.build_cc(sample, include_rank2=False)
            base_cc = histo_data.add_skeleton_graph(base_cc, sample)

            model_input = build_model_data(cc)
            cell_mask = cc.get_connected_nodes_mask()

            base_z, _ = model(*model_input)
            pred = torch.sigmoid(base_z)

            mask = (pred.squeeze()[model_input[-1][0]:-model_input[-1][2]] > thresh).detach().cpu().numpy()
            skeleton_to_trackster = torch.sparse.mm(cc.incidence_matrix(2, 1, weighted=False), cc.incidence_matrix(1, 3, weighted=False))
            trackster_to_linked = torch.sparse.mm(skeleton_to_trackster, cc.incidence_matrix(3, 1, weighted=False)).T
            trackster_to_linked._values().fill_(1)
            incidence = cc.incidence_matrix(1, 2, weighted=False)
            adaptions = (trackster_to_linked - incidence).to_dense()
            adaptions[mask] = 0
            print("num adaptions", adaptions.sum())

            children = ak.to_list(rank2_children)
            adap_coords = torch.nonzero(adaptions)
            associations = cc.incidence_matrix(0, 2, weighted=False).to_dense()[cell_mask]
            connections = torch.where(associations.any(dim=1), associations.ne(0).float().argmax(dim=1), torch.full((associations.shape[0],), -1, device=associations.device)).unsqueeze(-1)
            probs = pred

            for i, new_parent in adap_coords:
                new_parent = int(new_parent)

                old_parent = torch.nonzero(incidence.to_dense()[i])
                if (old_parent.shape[0] > 0):
                    old_parent = old_parent[0]

                    # remove i from second subarray of old parent
                    children[old_parent][1] = [
                        x for x in children[old_parent][1]
                        if x != i
                    ]


                # add i to second subarray of new parent
                if i not in children[new_parent][1]:
                    children[new_parent][1].append(i)
                
                new_children = ak.Array(children)
                adap_cc = copy.deepcopy(base_cc)
                adap_cc.append_cells(2, rank2_pos, new_children)

                z, _ = model(*build_model_data(adap_cc))
                probs = torch.cat([probs, torch.sigmoid(z)], axis=-1)
                associations = adap_cc.incidence_matrix(0, 2, weighted=False).to_dense()[cell_mask]
                connections = torch.cat([connections, torch.where(associations.any(dim=1), associations.ne(0).float().argmax(dim=1), torch.full((associations.shape[0],), -1, device=associations.device)).unsqueeze(-1)], axis=-1)

            P = sinkhorn_from_cost(-probs, eps=0.05)
            assigned_parent = P.argmax(dim=1)[:model_input[-1][0]]
            rows = torch.arange(assigned_parent.shape[0], device=assigned_parent.device)
            chosen_values = connections[rows, assigned_parent]

            associations = histo_data.get_associations(sample)
            base_res = validate_selection(connections[:, 0], associations, model_input[-1][2], cell_mask)
            adap_res = validate_selection(chosen_values, associations, model_input[-1][2], cell_mask)
            print(base_res, adap_res)
            
            base_vals.append(base_res)
            adap_vals.append(adap_res)
            diffs.append(adap_res - base_res)

            if sample > n_events:
                break

    plt.figure()
    plt.hist(base_vals, bins=30, alpha=0.5, label="base")
    plt.hist(adap_vals, bins=30, alpha=0.5, label="adapted")
    plt.xlabel("result")
    plt.ylabel("count")
    plt.title("Absolute result distribution")
    plt.legend()
    plt.savefig(osp.join(CONFIG.plots, experiment_name, f"{experiment_name}_abs.png"))

    plt.figure()
    plt.hist(diffs, bins=30)
    plt.axvline(0, linestyle="--")
    plt.xlabel("adapted result minus base result")
    plt.ylabel("count")
    plt.title("Adaptation effect")
    plt.savefig(osp.join(CONFIG.plots, experiment_name, f"{experiment_name}_rel.png"))
