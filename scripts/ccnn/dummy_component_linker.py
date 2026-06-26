
from scripts.CONFIG import CONFIG
from ticllearning.cclinking.component_classifier import CellClassifier

from ticllearning.cclinking.train import train_model

if __name__ == "__main__":
    data_info = "closeby_multi_0pu"
    experiment_name = "dummy_cell_linking"

    in_channels = [6, 16, 16]
    hidden_channels = 32
    num_classes = 1
    num_layer = 1

    model = CellClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layer,
        num_ranks=3,
        attention=True
    ).to(CONFIG.device)

    train_model(model, data_info, experiment_name, CONFIG, epochs=10, max_events=20, debug=True)
