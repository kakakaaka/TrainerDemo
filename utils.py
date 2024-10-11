# utils.py

import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    设置整个训练过程的随机种子，保证可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
