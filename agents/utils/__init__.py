from typing import List

import torch


def named_stack(tensors: List[torch.Tensor], name='N'):
    names = tensors[0].names
    return torch.stack([t.rename(None) for t in tensors]).rename(name, *names)


def named_gather(input: torch.Tensor, dim: torch.Tensor):
    return input.rename(None).gather(dim.rename(None)).rename(input.names())
