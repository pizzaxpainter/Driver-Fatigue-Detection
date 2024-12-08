from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.amp import autocast
import pandas as pd


def generate_classification_report(model, test_loader, device, classes=['neg', 'pos']):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, masks, labels in tqdm(test_loader, desc="Generating Classification Report", leave=True):
            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if "cuda" in device.type:
                with autocast(device.type):
                    outputs = model(inputs, img_mask=masks, seq_mask=masks)
            else:
                outputs = model(inputs, img_mask=masks, seq_mask=masks)

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=classes)
    cm = confusion_matrix(all_labels, all_preds)

    print("Classification Report:\n", report)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def generate_detailed_classification_report(model, test_loader, device, classes=['Alert', 'Drowsy']):
    """
    Generates and visualizes a detailed classification report for a given model on the test dataset.
    
    Specifically tailored for Driver Drowsiness Detection, this function evaluates the model's performance,
    prints the classification metrics, and visualizes them through various plots.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model for drowsiness detection.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to perform computations on ('cpu' or 'cuda').
        classes (list of str, optional): List of class names. Defaults to ['Alert', 'Drowsy'].
    """
    
    # Set the model to evaluation mode
    model.eval()
    
    # Lists to store true labels and model predictions
    true_labels = []
    predicted_labels = []
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate over the test data loader with a progress bar
        for inputs, masks, labels in tqdm(
            test_loader, 
            desc="Generating Detailed Classification Report", 
            leave=True
        ):
            # Move data to the specified device
            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Perform mixed-precision inference if using CUDA for efficiency
            if device.type == "cuda":
                with autocast(device.type):
                    outputs = model(inputs, img_mask=masks, seq_mask=masks)
            else:
                outputs = model(inputs, img_mask=masks, seq_mask=masks)
            
            # Get the predicted class with the highest score
            _, preds = torch.max(outputs, dim=1)
            
            # Append the true labels and predictions to the respective lists
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())
    
    # Generate the classification report as a dictionary
    report_dict = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=classes, 
        output_dict=True
    )
    
    # Convert the classification report to a pandas DataFrame for easier manipulation
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Print the classification report in a readable format
    print("=== Classification Report ===\n")
    print(classification_report(true_labels, predicted_labels, target_names=classes))
    
    # Define the metrics to visualize
    metrics = ['precision', 'recall', 'f1-score']
    
    # Reshape the DataFrame for seaborn to plot metrics side by side
    metrics_df = report_df.loc[classes, metrics].reset_index().melt(
        id_vars='index', 
        value_vars=metrics, 
        var_name='Metric', 
        value_name='Score'
    )
    metrics_df = metrics_df.rename(columns={'index': 'Class'})
    
    # Plot Precision, Recall, and F1-Score side by side for each class
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Class', 
        y='Score', 
        hue='Metric', 
        data=metrics_df, 
        palette='Set2'
    )
    plt.title("Precision, Recall, and F1-Score per Class")
    plt.xlabel("Drowsiness State")
    plt.ylabel("Score")
    plt.ylim(0, 1)  # Metrics range between 0 and 1
    plt.legend(title='Metric')
    plt.show()
    
    # Plot Support (number of instances) for each class
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=report_df.index[:-3], 
        y=report_df['support'][:-3], 
        palette='viridis'
    )
    plt.title("Number of Instances per Drowsiness State")
    plt.xlabel("Drowsiness State")
    plt.ylabel("Number of Samples")
    plt.show()
    
    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        xticklabels=classes, 
        yticklabels=classes, 
        cmap='OrRd'
    )
    plt.title('Confusion Matrix for Driver Drowsiness Detection')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()