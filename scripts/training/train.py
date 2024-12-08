import os
import torch
from tqdm import tqdm
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import GradScaler, autocast
import copy
import pandas as pd
import gc

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

def train_one_epoch(
    model, train_loader, criterion, optimizer, device, scaler,
    adversary=None, max_grad_norm=1.0
):
    """
    Trains the model for one epoch, optionally using adversarial training.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
        scaler (GradScaler): Gradient scaler for mixed precision.
        adversary (AdversarialAttack, optional): Adversary for adversarial training.
        max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 1.0.

    Returns:
        tuple: (epoch_loss, epoch_accuracy, epoch_f1, lr_list)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    lr_list = []

    with tqdm(train_loader, desc="Training", leave=False) as pbar:
        for batch_inputs, batch_masks, batch_labels in pbar:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Standard forward pass
            if scaler is not None:
                with torch.amp.autocast(device.type):
                    batch_outputs = model(batch_inputs, img_mask=batch_masks, seq_mask=batch_masks)
                    loss = criterion(batch_outputs, batch_labels)
            else:
                batch_outputs = model(batch_inputs, img_mask=batch_masks, seq_mask=batch_masks)
                loss = criterion(batch_outputs, batch_labels)

            # Adversarial training
            if adversary is not None:
                if adversary.attack_type == 'awp':
                    # Perform AWP attack by passing batch_loss
                    loss_adv = adversary.generate(batch_loss=loss)
                    # Combine losses
                    total_loss = (loss + loss_adv) / 2
                else:
                    # Generate adversarial examples for input attacks
                    batch_inputs_adv = adversary.generate(inputs=batch_inputs, labels=batch_labels, masks=batch_masks)
                    batch_inputs_adv = batch_inputs_adv.to(device)

                    # Forward pass with adversarial examples
                    if scaler is not None:
                        with torch.amp.autocast(device.type):
                            batch_outputs_adv = model(batch_inputs_adv, img_mask=batch_masks, seq_mask=batch_masks)
                            loss_adv = criterion(batch_outputs_adv, batch_labels)
                    else:
                        batch_outputs_adv = model(batch_inputs_adv, img_mask=batch_masks, seq_mask=batch_masks)
                        loss_adv = criterion(batch_outputs_adv, batch_labels)

                    # Combine losses
                    total_loss = (loss + loss_adv) / 2

                # Backpropagation with adversarial loss
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
            else:
                # Backpropagation with standard loss
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Gradient clipping and optimizer step
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            # Record learning rate
            current_lr = optimizer.param_groups[0]['lr']
            lr_list.append(current_lr)

            # Update metrics
            running_loss += (total_loss.item() if adversary else loss.item()) * batch_inputs.size(0)
            _, batch_preds = torch.max(batch_outputs, 1)
            correct += (batch_preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            all_preds.extend(batch_preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            batch_f1 = f1_score(all_labels, all_preds, average="macro")
            batch_accuracy = 100 * correct / total
            pbar.set_postfix({'Loss': f"{running_loss/total:.4f}", 'F1': f"{batch_f1:.4f}", 'Acc': f"{batch_accuracy:.2f}"})

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_accuracy, epoch_f1, lr_list


def evaluate(model, data_loader, criterion, device, classes=['neg', 'pos'], mode='Validation'):
    """
    Evaluates the model on a given dataset and prints classification metrics.

    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for the dataset to evaluate.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        classes (list, optional): List of class names. Defaults to ['neg', 'pos'].
        mode (str, optional): Mode name for display purposes. Defaults to 'Validation'.

    Returns:
        dict: Dictionary containing loss, accuracy, and F1-score.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(data_loader, desc=f"{mode} Evaluation", leave=True) as pbar:
            for inputs, masks, labels in pbar:
                inputs = inputs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Mixed precision inference
                if "cuda" in device.type and torch.cuda.is_available():
                    with torch.amp.autocast(device.type):
                        outputs = model(inputs, img_mask=masks, seq_mask=masks)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs, img_mask=masks, seq_mask=masks)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if total > 0:
                    batch_f1 = f1_score(all_labels, all_preds, average="macro")
                    batch_accuracy = 100 * correct / total
                    pbar.set_postfix({'Loss': f"{running_loss/total:.4f}", 'F1': f"{batch_f1:.4f}", 'Acc': f"{batch_accuracy:.2f}"})

    # Final metrics
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    print(f"{mode} Loss: {avg_loss:.4f} | {mode} Acc: {accuracy:.2f}% | {mode} F1: {f1:.4f}")
    
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=classes)
    print(f"Classification Report ({mode}):\n{report}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({mode})')
    plt.savefig(f'confusion_matrix_{mode}.png')  # Save the figure instead of showing
    plt.close()  # Close the figure to free memory
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_score': f1
    }

def train(
    model, 
    train_loader, 
    val_loader, 
    test_loader, 
    criterion, 
    optimizer, 
    scheduler, 
    device, 
    num_epochs, 
    patience=10,
    checkpoint_dir='checkpoints',
    save_every=1,
    adversary=None
):
    """
    Orchestrates the training, validation, and test process over multiple epochs.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Device to train on.
        num_epochs (int): Number of training epochs.
        patience (int, optional): Early stopping patience. Defaults to 10.
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to 'checkpoints'.
        save_every (int, optional): Save a checkpoint every 'save_every' epochs. Defaults to 1.
        adversary (AdversarialAttack, optional): Adversary for adversarial training.

    Returns:
        pd.DataFrame: DataFrame containing training history.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    scaler = torch.amp.GradScaler() if device.type == 'cuda' and torch.cuda.is_available() else None

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "test_accuracy": [],
        "train_f1": [],
        "val_f1": [],
        "test_f1": [],
        "epoch_time": [],
        "gpu_memory_MB": [],
        "learning_rates": [],  # Store learning rates
    }
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    if device.type == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Training Phase
        train_loss, train_acc, train_f1, lr_list = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            adversary=adversary,  # Pass the adversary
            max_grad_norm=1.0
        )

        # Store the average learning rate from this epoch
        if lr_list:
            avg_lr = sum(lr_list) / len(lr_list)
            history["learning_rates"].append(avg_lr)
        else:
            history["learning_rates"].append(optimizer.param_groups[0]['lr'])

        # Validation Phase
        val_metrics = evaluate(
            model, val_loader, criterion, device, mode='Validation'
        )
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        val_f1 = val_metrics['f1_score']

        # Test Phase
        test_metrics = evaluate(
            model, test_loader, criterion, device, mode='Test'
        )
        test_loss = test_metrics['loss']
        test_accuracy = test_metrics['accuracy']
        test_f1 = test_metrics['f1_score']

        # Calculate epoch time and GPU memory usage
        epoch_time = time.time() - start_time
        memory_usage = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" and torch.cuda.is_available() else 0.0

        # Store metrics
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_accuracy)
        history["test_accuracy"].append(test_accuracy)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["test_f1"].append(test_f1)
        history["epoch_time"].append(epoch_time)
        history["gpu_memory_MB"].append(memory_usage)

        print(
            f"Epoch {epoch} Summary:\n"
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f}\n"
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Val F1: {val_f1:.4f}\n"
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}% | Test F1: {test_f1:.4f}\n"
            f"Epoch Time: {epoch_time:.2f}s | GPU Memory Usage: {memory_usage:.2f} MB\n"
        )

        # Scheduler step based on validation loss
        scheduler.step(val_loss)

        # Checkpointing
        if epoch % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            # Save the best model
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss decreased. Best model saved at {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Convert history to DataFrame for analysis
    history_df = pd.DataFrame(history)
    return history_df