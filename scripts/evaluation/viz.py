import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(history_df):
    """
    Plots training and evaluation metrics over epochs, including learning rates.
    
    Args:
        history_df (pd.DataFrame): DataFrame containing training history.
    """
    # Validate required columns
    required_columns = [
        'epoch', 'train_loss', 'val_loss', 'test_loss',
        'train_accuracy', 'val_accuracy', 'test_accuracy',
        'train_f1', 'val_f1', 'test_f1', 'epoch_time', 'gpu_memory_MB', 'learning_rate'
    ]
    for col in required_columns:
        if col not in history_df.columns:
            print(f"Warning: Missing column '{col}' in history DataFrame.")
            # Continue without raising an error
            # Adjust plotting accordingly
            # raise ValueError(f"Missing required column in history DataFrame: {col}")
    
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle("Training History", fontsize=20)
    
    # Loss
    axes[0, 0].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', marker='o')
    axes[0, 0].plot(history_df['epoch'], history_df['test_loss'], label='Test Loss', marker='o')
    axes[0, 0].set_title('Loss Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history_df['epoch'], history_df['train_accuracy'], label='Train Accuracy', marker='o')
    axes[0, 1].plot(history_df['epoch'], history_df['val_accuracy'], label='Val Accuracy', marker='o')
    axes[0, 1].plot(history_df['epoch'], history_df['test_accuracy'], label='Test Accuracy', marker='o')
    axes[0, 1].set_title('Accuracy Over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    
    # F1-Score
    axes[1, 0].plot(history_df['epoch'], history_df['train_f1'], label='Train F1-Score', marker='o')
    axes[1, 0].plot(history_df['epoch'], history_df['val_f1'], label='Val F1-Score', marker='o')
    axes[1, 0].plot(history_df['epoch'], history_df['test_f1'], label='Test F1-Score', marker='o')
    axes[1, 0].set_title('F1-Score Over Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].legend()
    
    # Epoch Time
    axes[1, 1].plot(history_df['epoch'], history_df['epoch_time'], label='Epoch Time', marker='o', color='green')
    axes[1, 1].set_title('Epoch Time Over Epochs')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].legend()
    
    # GPU Memory
    axes[2, 0].plot(history_df['epoch'], history_df['gpu_memory_MB'], label='GPU Memory Usage', marker='o', color='red')
    axes[2, 0].set_title('GPU Memory Usage Over Epochs')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Memory (MB)')
    axes[2, 0].legend()
    
    # Learning Rate
    if 'learning_rate' in history_df.columns:
        axes[2, 1].plot(history_df['epoch'], history_df['learning_rate'], label='Learning Rate', marker='o', color='purple')
        axes[2, 1].set_title('Learning Rate per Epoch')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Learning Rate')
        axes[2, 1].legend()
    else:
        # If learning rate is recorded per batch in 'learning_rates' column
        if 'learning_rates' in history_df.columns:
            learning_rates = history_df['learning_rates'].explode().reset_index(drop=True)
            steps = range(1, len(learning_rates) + 1)
            axes[2, 1].plot(steps, learning_rates, label='Learning Rate', color='purple')
            axes[2, 1].set_title('Learning Rate Schedule')
            axes[2, 1].set_xlabel('Training Steps')
            axes[2, 1].set_ylabel('Learning Rate')
            axes[2, 1].legend()
        else:
            # Hide unused subplot if learning rate data is not available
            axes[2, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()