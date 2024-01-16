import numpy as np
import torch
from models.CNN_task_2 import ensemble_task_2
from collections import OrderedDict
from torch.utils.data import DataLoader

def train_model(model : torch.nn.Module | ensemble_task_2, train_loader : DataLoader, val_loader : DataLoader,
                loss : torch.nn.modules.loss._Loss, device : torch.device, max_epochs : int, 
                learning_rate : float = 0.001, epoch_print_ratio : int = 10, early_stopping : bool = True,
                patience : int = 100, is_adam : bool = True, momentum : float = 0.0) -> tuple[OrderedDict, list[float], list[float], list[float], list[float]]:
    """
        This function trains the passed model up to the number of epochs specified and then returns *the state_dict()* of the model with the lowest loss on the validation set, as well as the training and validation losses and accuracies.
    
        The original model passed to this function is modified in-place. This means that the model passed is the one trained up to the number of epochs, while the returned model is the one with the lowest validation loss.
    """
    epoch_print = max_epochs//epoch_print_ratio
    best_validation_loss = np.inf
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_model_state_dict = model.state_dict()

    epochs_without_improvement = 0

    if isinstance(model, ensemble_task_2):
        optimizer = [torch.optim.Adam(model_voter.parameters(), lr=learning_rate) if is_adam else torch.optim.SGD(model_voter.parameters(), lr=learning_rate, momentum=momentum) for model_voter in model.models]
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if is_adam else torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(max_epochs):
        # Training
        if epoch % epoch_print == 0:
            print(f"Epoch {epoch + 1} of {max_epochs}")

        if isinstance(model, ensemble_task_2):
            for n, model_voter in enumerate(model.models):
                _train_individual_model(model_voter, train_loader, optimizer[n], loss, device)
        else:
            _train_individual_model(model, train_loader, optimizer, loss, device)

        train_loss, accuracy = _calculate_loss_accuracy(model, train_loader, loss, device)
        train_losses.append(train_loss)
        train_accuracies.append(accuracy)
        
        if epoch % epoch_print == 0:
            print(f"Training loss: {train_loss}")
            print(f"Training accuracy: {accuracy*100}%")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = _calculate_loss_accuracy(model, val_loader, loss, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            if epoch % epoch_print == 0:
                print(f"Validation loss: {val_loss}")
                print(f"Validation accuracy: {val_acc*100}%")
            if val_loss < best_validation_loss:
                print(f"Found new best model, validation loss: {val_loss}, validation accuracy: {val_acc*100}%")
                best_model_state_dict = model.state_dict()
                best_validation_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                # Early stopping if validation loss has not improved for the last patience epochs.
                if early_stopping and epochs_without_improvement >= patience:
                    print(f'Stopping training due to lack of improvement in last {patience} epochs.')
                    return (best_model_state_dict, train_losses, val_losses, train_accuracies, val_accuracies)

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

def _train_individual_model(model : torch.nn.Module, train_loader : DataLoader, optimizer : torch.optim.Optimizer,
                            loss : torch.nn.modules.loss._Loss, device : torch.device) -> None:
    model.train()
    train_loss = 0
    for x, y in iter(train_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        l = loss(y_pred, y)
        l.backward()
        optimizer.step()
        train_loss += l.item()
    return

def _calculate_loss_accuracy(model : torch.nn.Module, loader : DataLoader, loss : torch.nn.modules.loss._Loss, device : torch.device) -> tuple[float, float]:
    model.eval()
    loss_ = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in iter(loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            l = loss(y_pred, y)
            loss_ += l.item()
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    loss_ /= len(loader)
    accuracy = correct / total
    return (loss_, accuracy)