import torch

from agents.models.cnn import FullyCNN


def test_fully_cnn():
    x = torch.rand(2, 3, 96, 96, names=('N', 'C', 'H', 'W'))
    model = FullyCNN(num_actions=3)
    out = model(x)
    assert out.shape[1] == 3
    assert out.shape[0] == 2
