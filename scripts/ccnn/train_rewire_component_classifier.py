import torch
import os.path as osp

from scripts.CONFIG import CONFIG
from ticllearning.cclinking.component_classifier import CellClassifier

from ticllearning.cclinking.train import train_model

if __name__ == "__main__":
    data_info = "closeby_multi_0pu"
    experiment_name = "laplacian_build_cell_linking"
    retrain = True

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

    if retrain:
        date = "2026-06-16"
        extra_info = "epoch_3_dict"
        run_name = f"{date}_{experiment_name}_{data_info}"
        weights = torch.load(osp.join(CONFIG.model, run_name, f"{run_name}_{extra_info}.pt"), weights_only=True)
        model.load_state_dict(weights["model_state_dict"], strict=False)
        start_epoch = weights["epoch"]
    else:
        start_epoch = 0

    train_model(model, data_info, experiment_name, CONFIG, epochs=20, debug=True, start_epoch=start_epoch+1)
