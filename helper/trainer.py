import numpy as np
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader

def train_model(model : torch.nn.Module, train_loader : DataLoader, val_loader : DataLoader, loss : torch.nn.modules.loss._Loss, optimizer : torch.optim.Optimizer,
                 device : torch.device, epochs : int, epoch_print_ratio : int = 10) -> tuple[OrderedDict, list[float], list[float], list[float], list[float]]:
    """
        This function trains the passed model up to the number of epochs specified and then returns *the state_dict()* of the model with the lowest loss on the validation set, as well as the training and validation losses and accuracies.
    
        The original model passed to this function is modified in-place. This means that the model passed is the one trained up to the number of epochs, while the returned model is the one with the lowest validation loss.
    """
    epoch_print = epochs//epoch_print_ratio
    best_validation_loss = np.inf
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_model_state_dict = model.state_dict()

    # Training loop
    for epoch in range(epochs):
        if epoch % epoch_print == 0:
            print(f"Epoch {epoch + 1} of {epochs}")
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for x, y in iter(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
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
        if epoch % epoch_print == 0:
            print(f"Training loss: {train_loss}")
            print(f"Training accuracy: {accuracy*100}%")

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in iter(val_loader):
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_pred_val = model(x_val)
                l_val = loss(y_pred_val, y_val)
                val_loss += l_val.item()
                _, predicted = torch.max(y_pred_val.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            val_acc = correct / total
            val_accuracies.append(val_acc)
            if epoch % epoch_print == 0:
                print(f"Validation loss: {val_loss}")
                print(f"Validation accuracy: {val_acc*100}%")
            if val_loss < best_validation_loss:
                print(f"Found new best model, validation loss: {val_loss}, validation accuracy: {val_acc*100}%")
                best_model_state_dict = model.state_dict()
                best_validation_loss = val_loss

    return (best_model_state_dict, train_losses, val_losses, train_accuracies, val_accuracies)

def test_model(model : torch.nn.Module, test_loader : DataLoader, device : torch.device) -> float:

    total = 0
    correct = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_pred_test = model(x_test)
            _, predicted = torch.max(y_pred_test.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

    test_accuracy = 100 * correct / total
    return test_accuracy