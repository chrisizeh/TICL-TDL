import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import re

import uproot as uproot
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import awkward as ak

from concurrent.futures import ProcessPoolExecutor, as_completed

from awkward_complex.classes.spectral import Spectral
from awkward_complex.datasets.cern.build import CERN

import warnings
warnings.filterwarnings("ignore")

class CCData(Data):
    def __init__(self, file, event, x, L, A, ranks, y, num_cells):
        super().__init__()
        self.x = x
        self.L = L
        self.A = A
        self.ranks = ranks
        self.y = y
        self.num_cells = num_cells
        self.event = event
        self.file = file

def process_event(idx, sample, histo_data, dataset_dir):
    try:
        cc = histo_data.build_cc(sample)
        cc = histo_data.add_skeleton_graph(cc, sample)

        # TODO: Work also with isolated cluster
        _, L_adj, _ = Spectral.full_graded_laplacian(cc)
        cell_mask = torch.ones(L_adj.shape[0]).bool()
        cell_mask[:cc.num_nodes] = cc.get_connected_nodes_mask()

        x = cc.nodes
        x = x[cell_mask[:cc.num_nodes], :]
        num_cells = torch.Tensor([x.shape[0], cc._num_cells_at_rank(1), cc._num_cells_at_rank(2), cc._num_cells_at_rank(3)]).long()

        L_adj = Spectral.normalize_matrix(L_adj)
        L_adj = L_adj[cell_mask][:, cell_mask]
        L_adj = L_adj.to_dense()[:-cc._num_cells_at_rank(3), :-cc._num_cells_at_rank(3)]

        A = Spectral.graded_incidence_matrix(cc, weighted=True).to_dense()
        A = A[cell_mask][:, cell_mask]

        assoc = histo_data.get_associations(sample)
        y = torch.zeros_like(assoc)

        y[:cc.num_nodes] = (cc.incidence_matrix(0, 2, weighted=False) @ assoc[-num_cells[2]:]) == assoc[:cc.num_nodes]
        y[cc.num_nodes:-num_cells[2]] = (cc.incidence_matrix(1, 2, weighted=False) @ assoc[-num_cells[2]:]) == assoc[cc.num_nodes:-num_cells[2]]
        y = y[cell_mask[:-num_cells[3]]]
        y = y[:-num_cells[2]]
        y = y.unsqueeze(1)

        ranks = torch.cat([torch.zeros(cc.num_nodes), ak.to_torch(cc.cells.rank)]).to(cc.device)
        ranks = ranks[cell_mask]
        x = [x, ak.to_torch(cc.cells.features[cc.cells.rank == 1]).float().to(cc.device), ak.to_torch(cc.cells.features[cc.cells.rank == 2]).float().to(cc.device)]

        data = CCData(histo_data.file, sample, x, L_adj.to_sparse(), A.to_sparse(), ranks, y, num_cells)
        torch.save(data, osp.join(dataset_dir, f'data_{(idx+sample):05d}.pt'))
        return [torch.max(torch.abs(data.x[i]), axis=0).values.detach().cpu().numpy() for i in range(3)]
    except Exception as e:
        print(e)
        return None

class CCDataset(Dataset):
    #node_feature_keys = ["barycenter_eta", "barycenter_phi", "barycenter_z", "raw_energy"]
    #node_feature_dict = {k: v for v, k in enumerate(node_feature_keys)}
    #model_feature_keys = node_feature_keys

    def __init__(self, data_info, config, test=False, skeleton_features=False, node_scaler=None, num_workers=5, max_events=None):
        self.test = test
        self.skeleton_features = skeleton_features
        self.device = config.device
        self.num_workers = num_workers
        self.max_events = max_events

        self.data_info = data_info
        self.output_folder = osp.join(config.data, data_info)
        self.input_folder = osp.join(config.histo, data_info)
        os.makedirs(self.output_folder, exist_ok=True)

        if self.test:
            self.data_folder = osp.join(self.output_folder, "test")
        else:
            self.data_folder = osp.join(self.output_folder, "train")

        if (node_scaler is None and osp.isfile(osp.join(self.output_folder, "node_scaler.pt"))):
            self.node_scaler = torch.load(osp.join(self.output_folder, "node_scaler.pt"))
        else:
            self.node_scaler = node_scaler

        self.process()

    @property
    def processed_file_names(self):
        return sorted(glob(f"{self.data_folder}/data_*.pt"), key=os.path.basename) 

    def process(self):
        if osp.isfile(osp.join(self.data_folder, "DONE")):
            return

        if self.test:
            files = glob(f"{self.input_folder}/test/*.root")
        else:
            files = glob(f"{self.input_folder}/train/*.root")
        os.makedirs(self.data_folder, exist_ok=True)

        histo_data = CERN(files[0], self.device)
        remaining_events = len(files)*histo_data.n_events
        if self.max_events is not None:
            remaining_events = self.max_events

        idx = 0
        with tqdm(total=remaining_events) as pbar:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for file in files:
                    histo_data = CERN(file, self.device)

                    if (self.max_events is None or remaining_events >= histo_data.n_events):
                        futures = [executor.submit(process_event, idx, sample, histo_data, self.data_folder) for sample in range(histo_data.n_events)]
                        idx += histo_data.n_events
                    else:
                        #process_event(idx, 0, histo_data, self.data_folder)
                        futures = [executor.submit(process_event, idx, sample, histo_data, self.data_folder) for sample in range(remaining_events)]

                    remaining_events -=  histo_data.n_events
                    for future in as_completed(futures):
                        pbar.update(1)
                        max_features = future.result()
                        if (not self.test and max_features is not None):
                            if self.node_scaler is not None:
                                self.node_scaler = [torch.maximum(self.node_scaler[i], torch.tensor(max_features[i])) for i in range(3)]
                            else:
                                self.node_scaler = [torch.tensor(max_features[i]).float().to(self.device) for i in range(3)]

        if (not self.test):
            torch.save(self.node_scaler, osp.join(self.output_folder, "node_scaler.pt"))

        idx = 0
        for file in tqdm(self.processed_file_names, desc="Fixing holes"):
            if (int(re.findall(r'\d+', file)[-1]) != idx):
                sample = torch.load(file, weights_only=False)
                os.remove(file)
                torch.save(sample, osp.join(self.data_folder, f"data_{idx:05d}.pt"))
            idx += 1
        torch.save([], osp.join(self.data_folder, "DONE"))

    def __len__(self):
        return len(self.processed_file_names)

    def __iter__(self): 
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        return torch.load(osp.join(self.data_folder, f'data_{idx:05d}.pt'), weights_only=False)
