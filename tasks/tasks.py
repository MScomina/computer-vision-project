import torch
from torchvision import transforms

from models.CNN_task_1 import CNN_task_1
from models.CNN_task_2 import ensemble_task_2
from models.CNN_task_3 import CNN_task_3_classifier, CNN_task_3_svm

from utils.trainer import train_model, test_model
from utils.plotter import save_loss_accuracy_plot
from utils.dataset_composer import generate_dataloaders

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
_PATIENCE = 30

# Task 2
_CONV_KERNEL_SIZES = (3, 3, 3)
_PADDING = (1, 1, 1)
_BATCH_NORMALIZATION = True
_DROPOUT = 0.3
_NUMBER_OF_MODELS = 5
_BETA = 1.0
_CAP = 10.0

# Task 3
_EPOCHS_SVM = 3

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

    train_loader, val_loader, test_loader = \
        generate_dataloaders(
            "dataset", 
            transform, 
            split_ratio=SPLIT_RATIO, 
            batch_size=BATCH_SIZE
        )

    model = CNN_task_1().to(device)

    loss = torch.nn.CrossEntropyLoss()

    # Train and save model and training and validation losses and accuracies
    best_model, train_losses, val_losses, train_accuracies, val_accuracies = \
        train_model(
            model=model,
            train_loader=train_loader, 
            val_loader=val_loader, 
            loss=loss, 
            device=device, 
            max_epochs=MAX_EPOCHS, 
            learning_rate=LEARNING_RATE,
            epoch_print_ratio=EPOCH_PRINT_RATIO, 
            early_stopping=EARLY_STOPPING, 
            patience=PATIENCE,
            is_adam=False,
            momentum=MOMENTUM
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
            EARLY_STOPPING=_EARLY_STOPPING, PATIENCE=_PATIENCE, BATCH_NORMALIZATION=_BATCH_NORMALIZATION, NUMBER_OF_MODELS=_NUMBER_OF_MODELS,
            BETA=_BETA, CAP=_CAP):

    # Anisotropic scaling and conversion to tensor.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),                                            # Normalize image to [-1, 1]
    ])

    data_augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),                                                      # Randomly flip image horizontally
        transforms.RandomRotation(degrees=20),                                                  # Randomly rotate image
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
        transforms.Lambda(lambda x: torch.clamp(x + 0.02*torch.randn_like(x),min=0.,max=1.)),   # Add random noise to image.
        transforms.Normalize(mean=[0.5], std=[0.5])                                             # Normalize image to [-1, 1]
    ])

    train_loader, val_loader, test_loader = \
        generate_dataloaders(
            "dataset",
            (data_augmentation_transform, transform, transform),
            split_ratio=SPLIT_RATIO,
            batch_size=BATCH_SIZE
        )

    model = ensemble_task_2(
        number_of_models=NUMBER_OF_MODELS,
        conv_kernel_sizes=CONV_KERNEL_SIZES, 
        dropout=DROPOUT, 
        padding=PADDING, 
        batch_normalization=BATCH_NORMALIZATION,
        beta=BETA,
        cap=CAP
    ).to(device)

    loss = torch.nn.CrossEntropyLoss()

    # Train and save model and training and validation losses and accuracies
    best_model, train_losses, val_losses, train_accuracies, val_accuracies = \
        train_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            loss=loss,
            device=device, 
            max_epochs=MAX_EPOCHS, 
            learning_rate=LEARNING_RATE,
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






def task_3(SPLIT_RATIO=_SPLIT_RATIO, BATCH_SIZE=_BATCH_SIZE, LEARNING_RATE=_LEARNING_RATE, MOMENTUM=_MOMENTUM, MAX_EPOCHS=_MAX_EPOCHS,
            EPOCH_PRINT_RATIO=_EPOCH_PRINT_RATIO, EARLY_STOPPING=_EARLY_STOPPING, PATIENCE=_PATIENCE, EPOCHS_SVM=_EPOCHS_SVM):
    
    # Anisotropic scaling and conversion to tensor.
    # AlexNet expects 3-channel images and 224x224 images.
    # https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])             # Normalize image to ImageNet mean and std
    ])

    data_augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),                                                      # Randomly flip image horizontally
        transforms.RandomRotation(degrees=10),                                                  # Randomly rotate image
        transforms.RandomChoice([                                                               # Randomly choose whether apply anisotropic rescaling or random cropping (2 to 1 ratio).  
            transforms.Resize((224, 224)),
            transforms.Resize((224, 224)),
            transforms.Compose([
                transforms.RandomCrop(160),
                transforms.Resize(224)
            ])
        ]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.clamp(x + 0.01*torch.randn_like(x),min=0.,max=1.)),   # Add random noise to image.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])             # Normalize image to ImageNet mean and std
    ])

    train_loader, val_loader, test_loader = \
        generate_dataloaders(
            "dataset",
            (data_augmentation_transform, transform, transform),
            split_ratio=SPLIT_RATIO,
            batch_size=BATCH_SIZE
        )

    model = CNN_task_3_classifier().to(device)

    loss = torch.nn.CrossEntropyLoss()

    # Train and save model and training and validation losses and accuracies
    best_model, train_losses, val_losses, train_accuracies, val_accuracies = \
        train_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            loss=loss, 
            device=device, 
            max_epochs=MAX_EPOCHS, 
            learning_rate=LEARNING_RATE,
            epoch_print_ratio=EPOCH_PRINT_RATIO,
            early_stopping=EARLY_STOPPING,
            patience=PATIENCE
        )

    # Load best model
    model.load_state_dict(best_model)

    # We only have to save the last layer of the model, since the rest is frozen and can be loaded from torchvision.
    torch.save(model.alexnet.classifier[6].state_dict(), "models/model_task_3.pt")

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
        "plots/loss_and_accuracy_task_3_classifier.png"
    )

    # SVM model
    svm_model = CNN_task_3_svm(
        train_dataloader=train_loader, 
        path="models/model_task_3_svm",
        device=device,
        epochs=EPOCHS_SVM,
        linear_svm=True
    ).to(device)

    # Test accuracy
    test_accuracy = test_model(svm_model, test_loader, device)
    print(f"Test accuracy: {test_accuracy}%")

    svm_model.save_model("models/model_task_3_svm")