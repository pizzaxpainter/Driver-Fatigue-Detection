import torch
from tqdm import tqdm


def check_dataset_for_nans(dataset, num_samples=100):
    for i in tqdm(range(min(len(dataset), num_samples)), desc="Checking for NaNs in dataset"):
        images, mask, label = dataset[i]
        if torch.isnan(images).any() or torch.isinf(images).any():
            print(f"NaN or Inf found in images at index {i}")
            return False
        if torch.isnan(mask).any() or torch.isinf(mask).any():
            print(f"NaN or Inf found in mask at index {i}")
            return False
        # Labels are integers; typically, no NaNs or Infs
    print("No NaNs or Infs found in the dataset samples checked.")
    return True

def check_model_weights(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in parameter: {name}")
            return False
    print("No NaNs detected in model parameters.")
    return True

def calculate_model_size(model):
    """
    Calculate the size of a PyTorch model in terms of parameters and estimated memory usage.
    Uses exact element sizes instead of assumptions.
    
    Args:
        model: PyTorch model (nn.Module)
    
    Returns:
        dict: Dictionary containing parameter count and memory estimates
    """
    # Validate model has parameters
    assert len(list(model.parameters())) > 0, "The model has no parameters."
    
    # Calculate parameter size
    param_size = 0
    num_params = 0
    for param in model.parameters():
        num_params += param.nelement()
        param_size += param.nelement() * param.element_size()
    
    # Calculate buffer size
    buffer_size = 0
    if len(list(model.buffers())) == 0:
        print("The model has no buffers.")
    else:
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
    
    def format_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.3f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.3f} TB"
    
    total_size = param_size + buffer_size
    total_size_with_gradients = total_size * 2  # Account for gradients during training
    
    # Calculate size in MB for direct comparison
    size_mb = total_size / (1024**2)
    
    return {
        'total_params': num_params,
        'param_size': format_size(param_size),
        'buffer_size': format_size(buffer_size),
        'total_size': format_size(total_size),
        'total_size_with_gradients': format_size(total_size_with_gradients),
        'size_mb': f"{size_mb:.3f} MB"  # Added for direct comparison with original
    }