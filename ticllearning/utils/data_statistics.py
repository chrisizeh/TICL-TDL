import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[:n-1] = a[:n-1]
    ret[n - 1:] = ret[n - 1:] / n
    return ret


def plot_loss(train_loss_history, val_loss_history, ax=None, n=8, save=False, output_folder=None, filename=None):
    epochs = len(train_loss_history)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(6)
        fig.set_figwidth(8)

    if (n > 0):
        ax.plot(range(1, epochs+1), moving_average(train_loss_history, n=n), label='train', linewidth=2)
        ax.plot(range(1, epochs+1), moving_average(val_loss_history, n=n), label='val', linewidth=2)
    else:
        ax.plot(range(1, epochs+1), train_loss_history, label='train', linewidth=2)
        ax.plot(range(1, epochs+1), val_loss_history, label='val', linewidth=2)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.legend()

    if (save and output_folder is not None and filename is not None):
        path = os.path.join(output_folder, filename)
        plt.savefig(path)


def plot_data_distribution(X, keys):
    scols = int(np.ceil(len(keys)/2))
    srows = 2
    fig, axes = plt.subplots(scols, srows, figsize=(20, 35), constrained_layout=True)

    for i, key in enumerate(keys):
        ax_col = int(i % scols)
        ax_row = int(i/scols)

        sns.histplot(X[key], ax=axes[ax_col, ax_row], kde=True, stat="density", linewidth=0, bins=15)
        axes[ax_col, ax_row].set_title('Frequency distribution ' + key, fontsize=18)
        axes[ax_col, ax_row].set_xlabel(key, fontsize=15)
        axes[ax_col, ax_row].set_ylabel('Count', fontsize=15)

    fig.tight_layout()
    plt.show()
    return  

def accuracy_score(y, pred, weight):
    return ((y == pred).float() * weights).sum() / weights.sum().item()


def weighted_precision_recall_f1(y_true, y_pred, weights):
	# Ensure binary ints
    y_true = y_true.int()
    y_pred = y_pred.int()

    # Masks for positive class
    mask_pos = (y_true == 1)
    pred_pos = (y_pred == 1)

    # Weighted counts
    tp = torch.sum(weights[mask_pos & pred_pos])
    fp = torch.sum(weights[~mask_pos & pred_pos])
    fn = torch.sum(weights[mask_pos & ~pred_pos])
    tn = torch.sum(weights[~mask_pos & ~pred_pos])
    return weighted_precision_recall_f1_from_precalc(tp, fp, fn, tn)

def weighted_precision_recall_f1_from_precalc(tp, fp, fn, tn):
    # Precision
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = torch.tensor(0.0, dtype=torch.float32)

    # Recall
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = torch.tensor(0.0, dtype=torch.float32)

    # Specificity
    if tn + fp > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = torch.tensor(0.0, dtype=torch.float32)

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = torch.tensor(0.0, dtype=torch.float32)

    return precision.item(), recall.item(), specificity.item(), f1.item()
