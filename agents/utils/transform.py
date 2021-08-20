import numpy as np
import torch
from torchvision import transforms


def get_default_transform(device: torch.device):
    def numpy_img_to_tensor(a: np.ndarray):
        return torch.tensor(a, device=device, dtype=torch.float, names=('H', 'W', 'C')).align_to('C', 'H', 'W')

    return transforms.Compose(
        [lambda a: a.copy(), numpy_img_to_tensor,
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    )
