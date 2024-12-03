import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set the seed for reproducibility across various libraries and modules.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
    """
    # 1. Set the PYTHONHASHSEED environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Set the seed for Python's built-in random module
    random.seed(seed)

    # 3. Set the seed for NumPy
    np.random.seed(seed)

    # 4. Set the seed for PyTorch on CPU
    torch.manual_seed(seed)

    # 5. If using CUDA, set the seed for all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # # 6. Configure PyTorch to use deterministic algorithms
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    print(f"Seed set to {seed} for reproducibility.")

def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")