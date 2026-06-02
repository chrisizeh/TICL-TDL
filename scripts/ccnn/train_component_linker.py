from scripts.CONFIG import CONFIG
from ticllearning.cclinking.component_classifier import CellClassifier

from ticllearning.cclinking.train import train_model

if __name__ == "__main__":
    info = "multi_pion_close_train"

    in_channels = 4
    hidden_channels = 4
    num_classes = 1
    num_layer = 2

    model = CellClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        num_layers=num_layer,
    ).to(CONFIG.device)

    train_model(model, info, "trial", CONFIG, epochs=10)

