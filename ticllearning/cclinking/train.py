import os.path as osp
import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from awkward_complex.classes.spectral import Spectral
from awkward_complex.datasets.cern.build import CERN


from ticllearning.utils.training.loss_function import FocalLossLogits
from ticllearning.utils.training.save_model import save_model
from ticllearning.utils.data_statistics import plot_loss
from ticllearning.utils.plot_results import *
from ticllearning.utils.training.early_stopping import EarlyStopping

def train_epoch(epoch, model, data, loss_obj, optimizer, weighted=True):
    epoch_loss = 0

    model.train()
    step = 1
    last_loss = 0
    torch.autograd.set_detect_anomaly(True, check_nan=False)
    for sample in tqdm(range(data.n_events), desc=f"Training Epoch {epoch}"):
        # reset optimizer and enable training mode
        optimizer.zero_grad(set_to_none=True)

        cc = data.build_cc(sample)
        cc = data.add_skeleton_graph(cc, sample)
        x = cc.get_features()
        x = x[:-cc._num_cells_at_rank(3)]
        _, L_adj, _ = Spectral.full_graded_laplacian(cc)
        L_adj = L_adj.to_dense()[:-cc._num_cells_at_rank(3), :-cc._num_cells_at_rank(3)]
        L_adj[L_adj == 0] = 10e-8
        assoc = data.get_associations(sample)

        rank2_cells = cc._num_cells_at_rank(2)
        y = assoc
        y[:cc.num_nodes] = cc.incidence_matrix(0, 2) @ assoc[-rank2_cells:] == assoc[:cc.num_nodes]
        y[cc.num_nodes:-rank2_cells] = cc.incidence_matrix(1, 2) @ assoc[-rank2_cells:] == assoc[cc.num_nodes:-rank2_cells]
        y[-rank2_cells:] = 1
        y = y.unsqueeze(1)
        y = y[:-rank2_cells]

        z = model(x, L_adj.to_sparse())
        z = z[:-rank2_cells]
        
        # rescale weights to interval [0, 1]
        weights = x.clone().detach()[:-rank2_cells, -1]
        weights /= 300
        weights = torch.clamp(weights, 0.0, 1.0)
        weights = weights

        loss = loss_obj(z, y, weights)

        # back-propagate and update the weight
        if not torch.isfinite(loss): raise RuntimeError("Non-finite loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        # skip update if grad_norm is suspiciously large
        if (not torch.isfinite(grad_norm)) or (epoch > 5 and grad_norm > 1e4):
            print(f"[WARN] Bad grad norm {grad_norm} at step {step}, skipping update.")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        epoch_loss += loss.item()

        if step % 1000 == 0:
            last_loss = epoch_loss/step
            print(f"Step loss: {last_loss}")
        step += 1

    return float(epoch_loss)/step


def test_epoch(epoch, model, data, loss_obj, config, weighted=True, threshold=0.5):
    with torch.set_grad_enabled(False):
        model.eval()
        val_loss = 0.0

        # 0: tp, 1: fp, 2: fn, 3: tn
        stats = torch.zeros(4, device=config.device)
            
        for sample in tqdm(range(data.n_events), desc=f"Validation Epoch {epoch}"):
            cc = data.build_cc(sample)
            cc = data.add_skeleton_graph(cc, sample)
            x = cc.get_features()
            x = x[:-cc._num_cells_at_rank(3)]
            _, L_adj, _ = Spectral.full_graded_laplacian(cc)
            L_adj = L_adj.to_dense()[:-cc._num_cells_at_rank(3), :-cc._num_cells_at_rank(3)]
            L_adj[L_adj == 0] = 10e-8
            assoc = data.get_associations(sample)

            rank2_cells = cc._num_cells_at_rank(2)
            y = assoc
            y[:cc.num_nodes] = cc.incidence_matrix(0, 2) @ assoc[-rank2_cells:] == assoc[:cc.num_nodes]
            y[cc.num_nodes:-rank2_cells] = cc.incidence_matrix(1, 2) @ assoc[-rank2_cells:] == assoc[cc.num_nodes:-rank2_cells]
            y[-rank2_cells:] = 1
            y = y.unsqueeze(1)
            y = y[:-rank2_cells]

            z = model(x, L_adj.to_sparse())
            z = z[:-rank2_cells]

            y_pred = (z > threshold).squeeze()
            y_true = (y > 0).squeeze()
            
            # rescale weights to interval [0, 1]
            weights = x[:-rank2_cells, -1]

            stats[0] += torch.sum(weights * (y_true & y_pred)).item()
            stats[1] += torch.sum(weights * (~y_true & y_pred)).item()
            stats[2] += torch.sum(weights * (y_true & ~y_pred)).item()
            stats[3] += torch.sum(weights * (~y_true & ~y_pred)).item()
            
            # rescale weights to interval [0, 1]
            weights /= 300
            weights = torch.clamp(weights, 0.0, 1.0)
            weights = weights.detach()

            loss = loss_obj(z, y, weights).item()
            val_loss += loss

        val_loss /= data.n_events
        return val_loss, stats


def validate_epoch(epoch, model, data, loss_obj, config, weighted=True, threshold=0.5):
    with torch.set_grad_enabled(False):
        model.eval()
        val_loss = 0.0

        pred, ys, weights = [], [], []
            
        for sample in tqdm(range(data.n_events), desc=f"Validation Epoch {epoch}"):
            cc = data.build_cc(sample)
            cc = data.add_skeleton_graph(cc, sample)
            x = cc.get_features()
            x = x[:-cc._num_cells_at_rank(3)]
            _, L_adj, _ = Spectral.full_graded_laplacian(cc)
            L_adj = L_adj.to_dense()[:-cc._num_cells_at_rank(3), :-cc._num_cells_at_rank(3)]
            L_adj[L_adj == 0] = 10e-8
            assoc = data.get_associations(sample)

            rank2_cells = cc._num_cells_at_rank(2)
            y = assoc
            y[:cc.num_nodes] = cc.incidence_matrix(0, 2) @ assoc[-rank2_cells:] == assoc[:cc.num_nodes]
            y[cc.num_nodes:-rank2_cells] = cc.incidence_matrix(1, 2) @ assoc[-rank2_cells:] == assoc[cc.num_nodes:-rank2_cells]
            y[-rank2_cells:] = 1
            y = y.unsqueeze(1)
            y = y[:-rank2_cells]

            z = model(x, L_adj.to_sparse())
            z = z[:-rank2_cells]

            pred += z.tolist()
            ys += y.tolist()

            # rescale weights to interval [0, 1]
            weight = x[:-rank2_cells, -1]
            weights += weight.tolist()
            weight /= 300
            weight = torch.clamp(weight, 0.0, 1.0)
            weight = weight.detach()

            loss = loss_obj(z, y, weight).item()
            val_loss += loss

        val_loss /= data.n_events
    return val_loss, torch.Tensor(pred), torch.Tensor(ys), torch.Tensor(weights)

def train_model(model, dataset, experiment_name, config, start_epoch=0, epochs=100, threshold=0.5):
    date = f"{datetime.now():%Y-%m-%d}"
    run_name = f"{date}_{experiment_name}_{dataset}"

    model_folder = osp.join(config.model, run_name)
    plot_folder = osp.join(config.plots, run_name)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)
    data = CERN(config.data, dataset)

    # Prepare Model
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_obj = FocalLossLogits(alpha=0.7, gamma=2)
    early_stopping = EarlyStopping(patience=20, delta=0)
    scheduler = CosineAnnealingLR(optimizer, start_epoch+epochs, eta_min=1e-6)

    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(start_epoch, start_epoch+epochs):
        print(f'Epoch: {epoch}')
        loss = train_epoch(epoch, model, data, loss_obj, optimizer)
        train_loss_hist.append(loss)

        val_loss, stats = test_epoch(epoch, model, data, loss_obj, config, threshold=threshold)
        val_loss_hist.append(val_loss)
        print(f'Training loss: {loss}, Validation loss: {val_loss}, Learning Rate: {scheduler.get_last_lr()}')

        plot_loss(train_loss_hist, val_loss_hist, save=True, output_folder=plot_folder, filename=f"{run_name}_loss_epochs")

        print("Fast statistic on model threshold:")
        print_acc_scores_from_precalc(*stats)
        
        if ((epoch) % 10 == 0):
            print("Store Diagrams")

            val_loss, pred, y, weight = validate_epoch(epoch+1, model, data, loss_obj, config, threshold=threshold)

            print("weighted by raw energy:")
            plot_binned_validation_results(pred, y, weight, thres=threshold, output_folder=plot_folder, file_suffix=f"{run_name}_epoch_{epoch}")
            plot_validation_results(pred, y, save=True, output_folder=plot_folder, file_suffix=f"{run_name}_epoch_{epoch}", weight=weight)

        if ((epoch) % 5 == 0):
            print("Store Model")
            save_model(model, epoch, optimizer, train_loss_hist, val_loss_hist, output_folder=model_folder, filename=f"model_{date}")

        early_stopping(model, val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping after {epoch+1} epochs")
            early_stopping.load_best_model(model)

            save_model(model, epoch, optimizer, train_loss_hist, val_loss_hist, output_folder=model_folder, filename=f"{run_name}_final_loss_{-early_stopping.best_score:.4f}")
            break

        scheduler.step()
        plt.close()
