import torch
from torchvision import transforms, datasets

from models.CNN_task_1 import CNN_task_1
from models.CNN_task_2 import CNN_task_2

from helper.trainer import train_model, test_model
from helper.plotter import save_loss_accuracy_plot

# Debug constants
_EPOCH_PRINT_RATIO = 10

# Hyperparameters

# Imposed by the problem statement
_SPLIT_RATIO = 0.85
_BATCH_SIZE = 32

# Task 1
_LEARNING_RATE = 2e-3
_MOMENTUM = 0.9
_EPOCHS = 200

# Task 2
_CONV_KERNEL_SIZES = 3
_DROPOUT = 0.3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def task_1(SPLIT_RATIO=_SPLIT_RATIO, BATCH_SIZE=_BATCH_SIZE, LEARNING_RATE=_LEARNING_RATE, MOMENTUM=_MOMENTUM, EPOCHS=_EPOCHS, EPOCH_PRINT_RATIO=_EPOCH_PRINT_RATIO):

    # Anisotropic scaling and conversion to tensor.
    # Note: since ToTensor() converts the image to [0, 1], we need to multiply by 255 to get the original values, since the problem statement mandates no preprocessing.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
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

    model = CNN_task_1().to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Train and save model and training and validation losses and accuracies
    best_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, loss, optimizer, device, EPOCHS, epoch_print_ratio=EPOCH_PRINT_RATIO)

    # Load best model
    model.load_state_dict(best_model)
    torch.save(best_model, "models/model_task_1.pt")

    # Test model
    test_accuracy = test_model(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy}%")

    values_str = f'Split Ratio: {SPLIT_RATIO}   Batch Size: {BATCH_SIZE}\nLearning Rate: {LEARNING_RATE}   Momentum: {MOMENTUM}   Epochs: {EPOCHS}\n'

    # Save plot
    save_loss_accuracy_plot(train_losses, val_losses, train_accuracies, val_accuracies, test_accuracy, values_str, "plots/loss_and_accuracy_task_1.png")






def task_2(SPLIT_RATIO=_SPLIT_RATIO, BATCH_SIZE=_BATCH_SIZE, LEARNING_RATE=_LEARNING_RATE, MOMENTUM=_MOMENTUM, EPOCHS=_EPOCHS,
            EPOCH_PRINT_RATIO=_EPOCH_PRINT_RATIO, CONV_KERNEL_SIZES=_CONV_KERNEL_SIZES, DROPOUT=_DROPOUT):

    # Anisotropic scaling and conversion to tensor.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    data_augmentation_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Load datasets
    full_training_data = datasets.ImageFolder(root="dataset/train")
    test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)

    # Split training data into training and validation sets
    train_size = int(SPLIT_RATIO * len(full_training_data))
    val_size = len(full_training_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_training_data, [train_size, val_size])

    # Apply the transformations to the subsets of the data
    # This is required since the data augmentation transform should only be applied to the training data, and random_split doesn't allow different transforms for the subsets.
    # Get the image paths and labels for the training and validation datasets
    train_images = [full_training_data[i][0] for i in train_dataset.indices]
    train_labels = [full_training_data[i][1] for i in train_dataset.indices]
    val_images = [full_training_data[i][0] for i in val_dataset.indices]
    val_labels = [full_training_data[i][1] for i in val_dataset.indices]

    # Apply the transformations to the images
    train_images = [data_augmentation_transform(image) for image in train_images]
    val_images = [transform(image) for image in val_images]

    # Convert the lists to tensors
    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)
    val_images = torch.stack(val_images)
    val_labels = torch.tensor(val_labels)

    # Create TensorDataset instances
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN_task_2(conv_kernel_sizes=CONV_KERNEL_SIZES, dropout=DROPOUT).to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Train and save model and training and validation losses and accuracies
    best_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, loss, optimizer, device, EPOCHS, epoch_print_ratio=EPOCH_PRINT_RATIO)

    # Load best model
    model.load_state_dict(best_model)
    torch.save(best_model, "models/model_task_2.pt")

    # Test model
    test_accuracy = test_model(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy}%")

    values_str = f'Split Ratio: {SPLIT_RATIO}   Batch Size: {BATCH_SIZE}\nLearning Rate: {LEARNING_RATE}   Momentum: {MOMENTUM}   Epochs: {EPOCHS}\n'

    # Save plot
    save_loss_accuracy_plot(train_losses, val_losses, train_accuracies, val_accuracies, test_accuracy, values_str, "plots/loss_and_accuracy_task_2.png")