import matplotlib.pyplot as plt

def save_loss_accuracy_plot(train_losses : list[float], val_losses : list[float], train_accuracies : list[float], val_accuracies : list[float],
                             test_accuracy : float, values_str : str, save_path : str) -> None:
    # Plot training and validation loss
    plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Add the values as text above the plots
    plt.figtext(0.5, 0.99, values_str, ha="center", va="top", fontsize=12)

    # Add test accuracy as text
    plt.figtext(0.5, 0.01, f'Final Test Accuracy: {test_accuracy:.3f}%', ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    plt.savefig(save_path)