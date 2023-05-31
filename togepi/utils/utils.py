import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Setting `torch.backends.cudnn.benchmark = False` slows down training.
    # Reference: https://pytorch.org/docs/stable/notes/randomness.html.
    torch.backends.cudnn.benchmark = True


def set_precision():
    # https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
