import torch
from torchvision import transforms, datasets

from models.CNN_task_1 import CNN_task_1
from models.CNN_task_2 import ensemble_task_2

from helper.trainer import train_model, test_model
from helper.plotter import save_loss_accuracy_plot

# Debug constants
_EPOCH_PRINT_RATIO = 10

# Hyperparameters

# Imposed by the problem statement
_SPLIT_RATIO = 0.85
_BATCH_SIZE = 32

# Task 1
_LEARNING_RATE = 1e-3
_MOMENTUM = 0.9
_MAX_EPOCHS = 1000
_EARLY_STOPPING = True
_PATIENCE = 150

# Task 2
_CONV_KERNEL_SIZES = (3, 3, 3)
_PADDING = (1, 1, 1)
_BATCH_NORMALIZATION = True
_DROPOUT = 0.3
_NUMBER_OF_MODELS = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def task_1(SPLIT_RATIO=_SPLIT_RATIO, BATCH_SIZE=_BATCH_SIZE, LEARNING_RATE=_LEARNING_RATE, MOMENTUM=_MOMENTUM, MAX_EPOCHS=_MAX_EPOCHS, 
           EPOCH_PRINT_RATIO=_EPOCH_PRINT_RATIO, EARLY_STOPPING=_EARLY_STOPPING, PATIENCE=_PATIENCE):

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
    best_model, train_losses, val_losses, train_accuracies, val_accuracies = \
        train_model(
            model,
            train_loader, 
            val_loader, 
            loss, 
            optimizer, 
            device, 
            MAX_EPOCHS, 
            epoch_print_ratio=EPOCH_PRINT_RATIO, 
            early_stopping=EARLY_STOPPING, 
            patience=PATIENCE
        )

    # Load best model
    model.load_state_dict(best_model)
    torch.save(best_model, "models/model_task_1.pt")

    # Test model
    test_accuracy = test_model(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy}%")

    values_str = (
        f'Split Ratio: {SPLIT_RATIO}   Batch Size: {BATCH_SIZE}   Total Epochs: {len(val_accuracies)-1}\n'
        f'Learning Rate: {LEARNING_RATE}   Momentum: {MOMENTUM}   Max Epochs: {MAX_EPOCHS}\n'
    )

    # Save plot
    save_loss_accuracy_plot(
        train_losses, 
        val_losses, 
        train_accuracies, 
        val_accuracies, 
        test_accuracy, 
        values_str, 
        "plots/loss_and_accuracy_task_1.png"
    )






def task_2(SPLIT_RATIO=_SPLIT_RATIO, BATCH_SIZE=_BATCH_SIZE, LEARNING_RATE=_LEARNING_RATE, MOMENTUM=_MOMENTUM, MAX_EPOCHS=_MAX_EPOCHS,
            EPOCH_PRINT_RATIO=_EPOCH_PRINT_RATIO, CONV_KERNEL_SIZES=_CONV_KERNEL_SIZES, DROPOUT=_DROPOUT, PADDING=_PADDING,
            EARLY_STOPPING=_EARLY_STOPPING, PATIENCE=_PATIENCE, BATCH_NORMALIZATION=_BATCH_NORMALIZATION, NUMBER_OF_MODELS=_NUMBER_OF_MODELS):

    # Anisotropic scaling and conversion to tensor.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),                                            # Normalize image to [-1, 1]
    ])

    data_augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),                                                      # Randomly flip image horizontally
        transforms.RandomRotation(degrees=10),                                                  # Randomly rotate image
        transforms.RandomChoice([                                                               # Randomly choose whether apply anisotropic rescaling or random cropping (2 to 1 ratio).  
            transforms.Resize((64, 64)),
            transforms.Resize((64, 64)),
            transforms.Compose([
                transforms.RandomCrop(160),
                transforms.Resize(64)
            ])
        ]),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.clamp(x + 0.01*torch.randn_like(x),min=0.,max=1.)),   # Add random noise to image.
        transforms.Normalize(mean=[0.5], std=[0.5])                                             # Normalize image to [-1, 1]
    ])

    # Load datasets
    full_training_data = datasets.ImageFolder(root="dataset/train")
    test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)

    # Split training data into training and validation sets
    # Note: i couldn't rely on torch.utils.data.random_split() because it doesn't let you specify different transforms for each dataset.
    train_size = int(SPLIT_RATIO * len(full_training_data))
    indices = torch.randperm(len(full_training_data)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create new datasets with the desired transforms
    train_dataset = datasets.ImageFolder(root="dataset/train", transform=data_augmentation_transform)
    val_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)

    # Use the indices to get the corresponding data
    train_dataset.samples = [train_dataset.samples[i] for i in train_indices]
    train_dataset.targets = [train_dataset.targets[i] for i in train_indices]

    val_dataset.samples = [val_dataset.samples[i] for i in val_indices]
    val_dataset.targets = [val_dataset.targets[i] for i in val_indices]

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ensemble_task_2(
        number_of_models=NUMBER_OF_MODELS,
        conv_kernel_sizes=CONV_KERNEL_SIZES, 
        dropout=DROPOUT, 
        padding=PADDING, 
        batch_normalization=BATCH_NORMALIZATION
    ).to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train and save model and training and validation losses and accuracies
    best_model, train_losses, val_losses, train_accuracies, val_accuracies = \
        train_model(
            model, 
            train_loader, 
            val_loader, 
            loss, 
            optimizer, 
            device, 
            MAX_EPOCHS, 
            epoch_print_ratio=EPOCH_PRINT_RATIO,
            early_stopping=EARLY_STOPPING,
            patience=PATIENCE
        )

    # Load best model
    model.load_state_dict(best_model)
    torch.save(best_model, "models/model_task_2.pt")

    # Test model
    test_accuracy = test_model(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy}%")

    values_str = (
        f'Split Ratio: {SPLIT_RATIO}   Batch Size: {BATCH_SIZE}   Total Epochs: {len(val_accuracies)-1}\n'
        f'Learning Rate: {LEARNING_RATE}   Momentum: {MOMENTUM}   Max Epochs: {MAX_EPOCHS}\n'    
        f'Convolutional Kernel Sizes: {CONV_KERNEL_SIZES}    Dropout: {DROPOUT}   Padding: {PADDING}\n'
        f'Batch Normalization: {BATCH_NORMALIZATION}   Number of Models: {NUMBER_OF_MODELS}\n'
    )

    # Save plot
    save_loss_accuracy_plot(
        train_losses, 
        val_losses, 
        train_accuracies, 
        val_accuracies, 
        test_accuracy, 
        values_str, 
        "plots/loss_and_accuracy_task_2.png"
    )