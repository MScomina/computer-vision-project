import numpy as np
import torch
from torchvision import transforms, datasets
from model.CNN_task_1 import CNN_task_1
import matplotlib.pyplot as plt

_SPLIT_RATIO = 0.85
_BATCH_SIZE = 32

_LEARNING_RATE = 3e-2
_MOMENTUM = 0.9
_EPOCHS = 1500
_EPOCH_PRINT_RATIO = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def task_1(SPLIT_RATIO=_SPLIT_RATIO, BATCH_SIZE=_BATCH_SIZE, LEARNING_RATE=_LEARNING_RATE, MOMENTUM=_MOMENTUM, EPOCHS=_EPOCHS, EPOCH_PRINT_RATIO=_EPOCH_PRINT_RATIO):

    EPOCH_PRINT = EPOCHS//EPOCH_PRINT_RATIO

    # Anisotropic scaling and conversion to tensor:
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # Load datasets
    full_training_data = datasets.ImageFolder(root="dataset/train", transform=transform)
    test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)

    # Split training data into training and validation sets
    train_size = int(SPLIT_RATIO * len(full_training_data))
    val_size = len(full_training_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_training_data, [train_size, val_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model_task_1 = CNN_task_1(padding=1).to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_task_1.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    best_validation_loss = np.inf
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(EPOCHS):
        if epoch % EPOCH_PRINT == 0:
            print(f"Epoch {epoch + 1} of {EPOCHS}")
        model_task_1.train()
        train_loss = 0
        correct = 0
        total = 0
        for x, y in iter(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model_task_1(x)
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            # Calculate accuracy
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        accuracy = correct / total
        train_accuracies.append(accuracy)
        if epoch % EPOCH_PRINT == 0:
            print(f"Training loss: {train_loss}")
            print(f"Training accuracy: {accuracy*100}%")

        # Validation loop
        model_task_1.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in iter(val_loader):
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_pred_val = model_task_1(x_val)
                l_val = loss(y_pred_val, y_val)
                val_loss += l_val.item()
                _, predicted = torch.max(y_pred_val.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            val_acc = correct / total
            val_accuracies.append(val_acc)
            if epoch % EPOCH_PRINT == 0:
                print(f"Validation loss: {val_loss}")
                print(f"Validation accuracy: {val_acc*100}%")
            if val_loss < best_validation_loss:
                print(f"Saving new best model, validation loss: {val_loss}, validation accuracy: {val_acc*100}%")
                torch.save(model_task_1.state_dict(), "model/model_task_1.pt")
                best_validation_loss = val_loss

    # Load best model
    model_task_1.load_state_dict(torch.load("model/model_task_1.pt"))

    # Test loop
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_pred_test = model_task_1(x_test)
            _, predicted = torch.max(y_pred_test.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test accuracy: {test_accuracy}%")

    values_str = f'Split Ratio: {_SPLIT_RATIO}   Batch Size: {_BATCH_SIZE}\nLearning Rate: {_LEARNING_RATE}   Momentum: {_MOMENTUM}   Epochs: {_EPOCHS}\n'

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

    plt.savefig("plots/loss_and_accuracy_task_1.png")