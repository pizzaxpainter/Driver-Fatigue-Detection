from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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


def generate_detailed_classification_report(model, test_loader, device, classes=['neg', 'pos']):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, masks, labels in tqdm(test_loader, desc="Generating Detailed Classification Report", leave=True):
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

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=classes))

    # Plot Precision, Recall, and F1-Score
    metrics = ['precision', 'recall', 'f1-score']
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        sns.barplot(x=report_df.index[:-3], y=report_df[metric][:-3], label=metric.capitalize())
    plt.title("Classification Metrics")
    plt.xlabel("Classes")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    # Plot Support
    plt.figure(figsize=(6, 4))
    sns.barplot(x=report_df.index[:-3], y=report_df['support'][:-3], palette='viridis')
    plt.title("Support per Class")
    plt.xlabel("Classes")
    plt.ylabel("Number of Instances")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='OrRd')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()