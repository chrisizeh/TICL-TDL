import torch

class CONFIG:
    histo = "/data/czeh/data_tdl/histos"
    data = "/data/czeh/data_tdl/datasets"
    model = "/data/czeh/data_tdl/models"
    plots = "/data/czeh/data_tdl/plots"
    device = "cuda" if torch.cuda.is_available() else "cpu"
