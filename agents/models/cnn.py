import torch
from pytorch_named_dims import nm
from torch.nn import Module, Sequential


class FullyCNN(Module):
    def __init__(self, num_features: int = 16, num_actions: int = 3):
        super().__init__()
        body = [
            nm.Conv2d(3, num_features, 3),
            nm.ReLU(),
            nm.MaxPool2d(2),
            nm.Conv2d(num_features, num_features, 3),
            nm.ReLU(),
            nm.MaxPool2d(2),
            nm.Conv2d(num_features, num_features, 3),
            nm.ReLU(),
            nm.MaxPool2d(2),
            nm.Conv2d(num_features, num_features, 3),
            nm.ReLU(),
            nm.MaxPool2d(2),
            nm.Conv2d(num_features, num_actions, 3),
            nm.ReLU(),
            nm.AdaptiveAvgPool2d(1),
        ]
        self.body = Sequential(*body)

    def forward(self, x: torch.Tensor):
        return self.body(x).flatten(['C', 'H', 'W'], 'C')
