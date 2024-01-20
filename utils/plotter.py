import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

def save_loss_accuracy_plot(train_losses : list[float], val_losses : list[float], train_accuracies : list[float], 
                            val_accuracies : list[float], values_str : str, save_path : str) -> None:
    # Plot training and validation loss
    plt.figure(figsize=(16, 9))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.legend()

    #plt.figtext(0.5, 0.99, values_str, ha="center", va="top", fontsize=12)

    plt.subplots_adjust(top=0.95,left=0.05,right=0.95)
    plt.savefig(save_path)
    plt.close()

def save_confusion_matrix(confusion_matrix : list[list[int]], test_accuracy : float, save_path : str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.figtext(0.5, 0.01, f'Final Test Accuracy: {test_accuracy:.3f}%', ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.subplots_adjust(top=0.95,left=0.1,right=1.0)
    plt.savefig(save_path)
    plt.close()