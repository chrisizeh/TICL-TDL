import torch
from torch import nn
import torch.nn.functional as F

from topomodelx.base.message_passing import MessagePassing

class HasseMP(MessagePassing):
    def __init__(self, channels, max_rank):
        super().__init__(aggr_func="mean", att=False)
        self.max_rank = max_rank

        self.message_nn = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def message(self, x_source, x_target=None):
        return self.message_nn(x_source)
