import os.path as osp
import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader.dataloader import DataLoader
import matplotlib.pyplot as plt

from ticllearning.datasets.ccnn.dataset import CCDataset
from ticllearning.utils.training.loss_function import FocalLossLogits
from ticllearning.utils.training.save_model import save_model
from ticllearning.utils.data_statistics import plot_loss
from ticllearning.utils.plot_results import *
from ticllearning.utils.training.early_stopping import EarlyStopping
from ticllearning.utils.graph_utils import negative_edge_imbalance

def train_epoch(epoch, model, data, loss_obj, optimizer, weighted=True):
    epoch_loss = 0

    model.train()
    step = 1
    last_loss = 0
    torch.autograd.set_detect_anomaly(True, check_nan=False)
    for sample in tqdm(data, desc=f"Training Epoch {epoch}"):
        # reset optimizer and enable training mode
        optimizer.zero_grad(set_to_none=True)

        z = model(sample.x, sample.L)
        z = z[:-sample.num_rank2]
        
        # rescale weights to interval [0, 1]
        weights = sample.x.clone().detach()[:-sample.num_rank2, -1]
        weights /= 300
        weights = torch.clamp(weights, 0.0, 1.0)
        weights = weights

        loss = loss_obj(z, sample.y, weights)

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
            
        for sample in tqdm(data, desc=f"Test Epoch {epoch}"):
            z = model(sample.x, sample.L)
            z = z[:-sample.num_rank2]
            
            # rescale weights to interval [0, 1]

            y_pred = (z > threshold).squeeze()
            y_true = (sample.y > 0).squeeze()
            
            # rescale weights to interval [0, 1]
            weights = sample.x.clone().detach()[:-sample.num_rank2, -1]

            stats[0] += torch.sum(weights * (y_true & y_pred)).item()
            stats[1] += torch.sum(weights * (~y_true & y_pred)).item()
            stats[2] += torch.sum(weights * (y_true & ~y_pred)).item()
            stats[3] += torch.sum(weights * (~y_true & ~y_pred)).item()
            
            # rescale weights to interval [0, 1]
            weights /= 300
            weights = torch.clamp(weights, 0.0, 1.0)
            weights = weights.detach()

            loss = loss_obj(z, sample.y, weights).item()
            val_loss += loss

        val_loss /= len(data)
        return val_loss, stats


def validate_epoch(epoch, model, data, loss_obj, config, weighted=True, threshold=0.5):
    with torch.set_grad_enabled(False):
        model.eval()
        val_loss = 0.0

        pred, ys, weights = [], [], []
            
        for sample in tqdm(data, desc=f"Validation Epoch {epoch}"):
            z = model(sample.x, sample.L)
            z = z[:-sample.num_rank2]
            
            pred += z.squeeze(-1).tolist()
            ys += sample.y.squeeze(-1).tolist()

            # rescale weights to interval [0, 1]
            weight = sample.x.clone().detach()[:-sample.num_rank2, -1]
            weights += weight.tolist()
            weight /= 300
            weight = torch.clamp(weight, 0.0, 1.0)
            weight = weight.detach()

            loss = loss_obj(z, sample.y, weight).item()
            val_loss += loss

        val_loss /= len(data)
    return val_loss, torch.Tensor(pred), torch.Tensor(ys), torch.Tensor(weights)

def train_model(model, dataset, experiment_name, config, start_epoch=0, epochs=100, threshold=0.5, max_events=None):
    date = f"{datetime.now():%Y-%m-%d}"
    run_name = f"{date}_{experiment_name}_{dataset}"

    model_folder = osp.join(config.model, run_name)
    plot_folder = osp.join(config.plots, run_name)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    batch_size = 1
    train_dataset = CCDataset(dataset, config, test=False, max_events=max_events)
    test_events = int(max_events*0.2) if max_events is not None else None
    test_dataset = CCDataset(dataset, config, test=True, max_events=test_events, node_scaler=train_dataset.node_scaler)
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dl = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    # Prepare Model
    model = model.to(config.device)
    model.add_scaler(train_dataset.node_scaler)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    alpha = 0.5 + negative_edge_imbalance(train_dataset)/2
    print("alpha: ", alpha)
    loss_obj = FocalLossLogits(alpha=alpha, gamma=2)
    early_stopping = EarlyStopping(patience=20, delta=0)
    scheduler = CosineAnnealingLR(optimizer, start_epoch+epochs, eta_min=1e-6)

    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(start_epoch, start_epoch+epochs):
        print(f'Epoch: {epoch}')
        loss = train_epoch(epoch, model, train_dl, loss_obj, optimizer)
        train_loss_hist.append(loss)

        val_loss, stats = test_epoch(epoch, model, test_dl, loss_obj, config, threshold=threshold)
        val_loss_hist.append(val_loss)
        print(f'Training loss: {loss}, Validation loss: {val_loss}, Learning Rate: {scheduler.get_last_lr()}')

        plot_loss(train_loss_hist, val_loss_hist, save=True, output_folder=plot_folder, filename=f"{run_name}_loss_epochs")

        print("Fast statistic on model threshold:")
        print_acc_scores_from_precalc(*stats)
        
        if ((epoch) % 10 == 0 and epoch != 0):
            print("Store Diagrams")

            val_loss, pred, y, weight = validate_epoch(epoch, model, test_dl, loss_obj, config, threshold=threshold)

            print("weighted by raw energy:")
            plot_binned_validation_results(pred, y, weight, thres=threshold, output_folder=plot_folder, file_suffix=f"{run_name}_epoch_{epoch}")
            plot_validation_results(pred, y, save=True, output_folder=plot_folder, file_suffix=f"{run_name}_epoch_{epoch}", weight=weight)

        if ((epoch) % 10 == 0 and epoch != 0):
            print("Store Model")
            save_model(model, epoch, optimizer, train_loss_hist, val_loss_hist, output_folder=model_folder, filename=f"{run_name}")

        early_stopping(model, val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping after {epoch+1} epochs")
            early_stopping.load_best_model(model)

            save_model(model, epoch, optimizer, train_loss_hist, val_loss_hist, output_folder=model_folder, filename=f"{run_name}_final_loss_{-early_stopping.best_score:.4f}")
            break

        scheduler.step()
        plt.close()
