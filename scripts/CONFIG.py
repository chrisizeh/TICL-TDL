import torch

class CONFIG:
    histo = "../data/histos"
    data = "../data/datasets"
    model = "../data/models"
    plots = "../data/plots"
    device = "cuda" if torch.cuda.is_available() else "cpu"
