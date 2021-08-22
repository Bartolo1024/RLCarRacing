import torch
from pytorch_named_dims import nm
from torch.nn import Module, Sequential


class FullyCNN(Module):
    def __init__(self, num_features: int = 8, num_actions: int = 4):
        super().__init__()
        body = [
            nm.Conv2d(3, num_features, 7, padding=3, stride=3),
            nm.ReLU(),
            nm.MaxPool2d(2),
            nm.Conv2d(num_features, num_features * 2, 5, padding=2, stride=2),
            nm.ReLU(),
            nm.MaxPool2d(2),
            nm.Conv2d(num_features * 2, num_features * 4, 3, padding=1, stride=2),
            nm.ReLU(),
            nm.MaxPool2d(2),
        ]
        self.head = nm.Conv2d(num_features * 4, num_actions, 1, padding=0)
        self.body = Sequential(*body)

    def forward(self, x: torch.Tensor):
        x = self.body(x)
        x = self.head(x)
        return x.flatten(['C', 'H', 'W'], 'C')


class CNNFC(Module):
    def __init__(self, num_features: int = 8, num_actions: int = 4):
        super().__init__()
        conv_backbone = [
            nm.Conv2d(3, num_features, 7, padding=3, stride=3),
            nm.ReLU(),
            nm.MaxPool2d(2),
            nm.Conv2d(num_features, num_features, 5, padding=2, stride=2),
            nm.ReLU(),
            nm.MaxPool2d(2),
        ]
        self.conv_backbone = Sequential(*conv_backbone)
        fc_head = [
            nm.Linear(num_features * 16, num_features * 4),
            nm.ReLU(),
            nm.Linear(num_features * 4, num_actions)
        ]
        self.fc_head = Sequential(*fc_head)

    def forward(self, x: torch.Tensor):
        x = self.conv_backbone(x)
        x = x.flatten(['C', 'H', 'W'], 'C')
        x = self.fc_head(x)
        return x
