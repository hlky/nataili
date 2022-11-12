from functools import wraps
from typing import TypeVar

import torch

T = TypeVar("T")

def performance(function: T) -> T:
    @wraps(function)
    def wrapper(*args, **kwargs):
        return torch.cuda.amp.autocast()(torch.no_grad()(function))(*args, **kwargs)
    return wrapper
