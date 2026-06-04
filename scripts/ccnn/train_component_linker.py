from scripts.CONFIG import CONFIG
from ticllearning.cclinking.component_classifier import CellClassifier

from ticllearning.cclinking.train import train_model

if __name__ == "__main__":
    data_info = "closeby_multi_0pu"
    experiment_name = "cell_linking"

    in_channels = 4
    hidden_channels = 164
    num_classes = 1
    num_layer = 2

    model = CellClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layer,
    ).to(CONFIG.device)

    train_model(model, data_info, experiment_name, CONFIG, epochs=100, max_events=500)

