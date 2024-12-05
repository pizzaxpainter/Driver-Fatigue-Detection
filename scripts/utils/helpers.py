import os
import torch

def save_model(model, optimizer=None, scheduler=None, history=None, epoch=None, file_path=None):
    """
    Saves the model and optionally optimizer, scheduler, training history, and epoch to a file.
    
    Parameters:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer, optional): The optimizer to save.
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler to save.
        history (list, optional): The training/validation history to save.
        epoch (int, optional): The epoch number to save.
        file_path (str): Path to save the checkpoint file.
    """
    if model is None:
        raise ValueError("Model is required to save the checkpoint.")
    if file_path is None:
        raise ValueError("File path is required to save the checkpoint.")

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save either full checkpoint or just model weights
    if optimizer or scheduler or history or epoch is not None:
        checkpoint = {'model_state_dict': model.state_dict()}
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if history:
            checkpoint['history'] = history
        if epoch is not None:
            checkpoint['epoch'] = epoch
        torch.save(checkpoint, file_path)
    else:
        # Save only model weights
        torch.save(model.state_dict(), file_path)

    print(f"Checkpoint saved at {file_path}")

def load_model(model, device, file_path, optimizer=None, scheduler=None):
    """
    Loads the model and optionally optimizer, scheduler from a checkpoint file.
    
    Parameters:
        model (torch.nn.Module): The PyTorch model to load.
        device (torch.device): The device to map the model and states.
        file_path (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler to load state into.
    
    Returns:
        dict: The loaded checkpoint contents (if applicable), else an empty dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

    checkpoint = torch.load(file_path, map_location=device)

    # Detect if the file contains only model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint detected
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully.")

        # Load optimizer state if available
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded successfully.")

        # Load scheduler state if available
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded successfully.")

        return checkpoint
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        # # Unexpected dictionary structure
        # raise ValueError("Invalid checkpoint structure.")
    else:
        # Only model weights detected
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")