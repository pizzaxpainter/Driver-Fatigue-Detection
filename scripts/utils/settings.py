import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


# Set global seed for reproducibility
def set_global_seed(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False,
    use_cuda: bool = True,
    use_mps: bool = True
) -> None:
    """
    Set random seed for reproducibility across multiple libraries and platforms.
    
    Args:
        seed (int): Random seed value
        deterministic (bool): If True, ensures completely reproducible results
                            by disabling CUDNN non-deterministic algorithms
        benchmark (bool): If True, enables CUDNN benchmark mode for potentially
                         faster training (but non-deterministic)
        use_cuda (bool): Whether to set CUDA seeds
        use_mps (bool): Whether to set MPS seeds (for Apple Silicon)
    """
    # Python's random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if use_cuda and torch.cuda.is_available():
        # CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # CUDNN
        cudnn.deterministic = deterministic
        cudnn.benchmark = benchmark
        
        if deterministic:
            # Ensure deterministic behavior for CUDA convolution operations
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True)
    
    if use_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS (Apple Silicon)
        torch.mps.manual_seed(seed)
    
    print(f"Global seed set to {seed}")
    print(f"Deterministic mode: {deterministic}")
    print(f"CUDNN benchmark mode: {benchmark}")
    print(f"Device configuration:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  Current CUDA device: {torch.cuda.current_device()}")

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