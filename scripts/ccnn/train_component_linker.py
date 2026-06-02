
import torch

from awkward_complex.classes.spectral import Spectral
from awkward_complex.datasets.cern.build import CERN

from ticllearning.cclinking.component_classifier import CellClassifier
import torch.nn.functional as F


if __name__ == "__main__":
    base_folder = "../"
    info = "multi_pion_close_train"

    data = CERN(base_folder, info)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_channels = 4
    hidden_channels = 32
    num_classes = 1
    num_layer = 4
    runs = 10

    model = CellClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layer,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(runs):
        for event in range(data.n_events):
            model.train()
            optimizer.zero_grad()

            cc = data.build_cc(event)
            cc = data.add_skeleton_graph(cc, event)
            x = cc.get_features()
            x = x[:-cc._num_cells_at_rank(3)]
            _, L_adj, _ = Spectral.full_graded_laplacian(cc)
            L_adj = L_adj.to_dense()[:-cc._num_cells_at_rank(3), :-cc._num_cells_at_rank(3)]
            L_adj[L_adj == 0] = 10e-8
            assoc = data.get_associations(event)

            rank2_cells = cc._num_cells_at_rank(2)
            y = assoc
            y[:cc.num_nodes] = cc.incidence_matrix(0, 2) @ assoc[-rank2_cells:] == assoc[:cc.num_nodes]
            y[cc.num_nodes:-rank2_cells] = cc.incidence_matrix(1, 2) @ assoc[-rank2_cells:] == assoc[cc.num_nodes:-rank2_cells]
            y[-rank2_cells:] = 1
            y = y.unsqueeze(1)

            logits = model(x, L_adj.to_sparse())
            loss = F.binary_cross_entropy(logits[:-rank2_cells], y[:-rank2_cells])

            loss.backward()
            optimizer.step()

            if event % 1 == 0:
                pred = logits.argmax(dim=-1)
                acc = (pred == y).float().mean()
                print(f"run={i:03d} epoch={event:03d} loss={loss.item():.8f} acc={acc.item():.3f}")
    model.eval()

    with torch.no_grad():
        for event in range(data.n_events):
            model.train()
            optimizer.zero_grad()

            cc = data.build_cc(event)
            x = cc.get_features()
            x = x[:-cc._num_cells_at_rank(3)]
            _, L_adj, _ = Spectral.full_graded_laplacian(cc)
            L_adj = L_adj.to_dense()[:-cc._num_cells_at_rank(3), :-cc._num_cells_at_rank(3)]
            L_adj[L_adj == 0] = 10e-8
            assoc = data.get_associations(event)

            rank2_cells = cc._num_cells_at_rank(2)
            y = assoc
            y[:cc.num_nodes] = cc.incidence_matrix(0, 2) @ assoc[-rank2_cells:] == assoc[:cc.num_nodes]
            y[cc.num_nodes:-rank2_cells] = cc.incidence_matrix(1, 2) @ assoc[-rank2_cells:] == assoc[cc.num_nodes:-rank2_cells]
            y[-rank2_cells:] = 1
            y = y.unsqueeze(1)

            logits = model(x, L_adj.to_sparse())
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean()
            print(f"run={i:03d} epoch={event:03d} loss={loss.item():.4f} acc={acc.item():.3f}")

