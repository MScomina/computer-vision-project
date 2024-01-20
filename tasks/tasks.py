import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix

from models.CNN_task_1 import CNN_task_1
from models.CNN_task_2 import ensemble_task_2
from models.CNN_task_3 import CNN_task_3_classifier, CNN_task_3_svm

from utils.trainer import train_model, test_model
from utils.plotter import save_loss_accuracy_plot, save_confusion_matrix
from utils.dataset_composer import generate_dataloaders

# Debug constants
_EPOCH_PRINT_RATIO = 10

# Hyperparameters

# Imposed by the problem statement
_SPLIT_RATIO = 0.85
_BATCH_SIZE = 32

# General
_SAVE_MODELS = True
_LEARNING_RATE = 1e-3   
_MOMENTUM = 0.9
_MAX_EPOCHS = 1000
_EARLY_STOPPING = True
_PATIENCE = 30

# Task 1
_PATH_1_DETAILS = ""

# Task 2
_CONV_KERNEL_SIZES = (3, 3, 3)
_CONV_KERNEL_CHANNELS = (8, 16, 32)
_BATCH_NORMALIZATION = True
_DROPOUT = 0.2
_NUMBER_OF_MODELS = 10
_RANDOM_HORIZONTAL_FLIP = True
_RANDOM_ROTATION = True
_RANDOM_CROP = True
_RANDOM_NOISE = True
_RELU = False
_KAIMING_NORMAL = True
_PATH_2_DETAILS = ""

# Only relevant if _RELU = False for Swish activation function
_SWISH_BETA = 1.0

# Task 3
_LOAD_SVM = True
_SVM_C = 1.0
_SVM_TYPE = "linear"
_EPOCHS_SVM = 3
_PATH_3_DETAILS = ""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def task_1(SPLIT_RATIO=_SPLIT_RATIO, BATCH_SIZE=_BATCH_SIZE, LEARNING_RATE=_LEARNING_RATE, MOMENTUM=_MOMENTUM, MAX_EPOCHS=_MAX_EPOCHS, 
           EPOCH_PRINT_RATIO=_EPOCH_PRINT_RATIO, EARLY_STOPPING=_EARLY_STOPPING, PATIENCE=_PATIENCE, PATH_DETAILS=_PATH_1_DETAILS,
           SAVE_MODEL=_SAVE_MODELS) -> float:

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
    if SAVE_MODEL:
        torch.save(best_model, "models/model_task_1"+PATH_DETAILS+".pt")

    # Test model
    test_accuracy, predictions, labels = test_model(model, test_loader, device)
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
        values_str, 
        "plots/loss_and_accuracy_task_1"+PATH_DETAILS+".png"
    )

    # Save confusion matrix
    save_confusion_matrix(
        confusion_matrix(labels, predictions),
        test_accuracy,
        "plots/confusion_matrix_task_1"+PATH_DETAILS+".png"
    )

    return test_accuracy





def task_2(SPLIT_RATIO=_SPLIT_RATIO, BATCH_SIZE=_BATCH_SIZE, LEARNING_RATE=_LEARNING_RATE, MOMENTUM=_MOMENTUM, MAX_EPOCHS=_MAX_EPOCHS,
            EPOCH_PRINT_RATIO=_EPOCH_PRINT_RATIO, CONV_KERNEL_SIZES=_CONV_KERNEL_SIZES, CONV_KERNEL_CHANNELS=_CONV_KERNEL_CHANNELS, DROPOUT=_DROPOUT,
            EARLY_STOPPING=_EARLY_STOPPING, PATIENCE=_PATIENCE, BATCH_NORMALIZATION=_BATCH_NORMALIZATION, NUMBER_OF_MODELS=_NUMBER_OF_MODELS,
            SWISH_BETA=_SWISH_BETA, KAIMING_NORMAL=_KAIMING_NORMAL, RANDOM_HORIZONTAL_FLIP=_RANDOM_HORIZONTAL_FLIP, RANDOM_ROTATION=_RANDOM_ROTATION, 
            RANDOM_CROP=_RANDOM_CROP, RANDOM_NOISE=_RANDOM_NOISE, PATH_DETAILS=_PATH_2_DETAILS, RELU=_RELU, SAVE_MODEL=_SAVE_MODELS) -> float:

    # Anisotropic scaling and conversion to tensor.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),                                            # Normalize image to [-1, 1]
    ])

    data_augmentation_transform_array = [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])                                             # Normalize image to [-1, 1]
    ]

    # Data augmentation
    if RANDOM_CROP:         # Randomly choose whether apply anisotropic rescaling or random cropping (2 to 1 ratio). 
        data_augmentation_transform_array.insert(0, transforms.RandomChoice([
            transforms.Resize((64, 64)),
            transforms.Resize((64, 64)),
            transforms.Compose([
                transforms.RandomCrop(160),
                transforms.Resize(64)
            ])
        ]))
    else:                   # Anisotropic scaling
        data_augmentation_transform_array.insert(0, transforms.Resize((64, 64)))
    if RANDOM_ROTATION:     # Randomly rotate image
        data_augmentation_transform_array.insert(0, transforms.RandomRotation(degrees=20))
    if RANDOM_HORIZONTAL_FLIP:     # Randomly flip image horizontally
        data_augmentation_transform_array.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    if RANDOM_NOISE:        # Add random noise to image.
        data_augmentation_transform_array.insert(-1, transforms.Lambda(lambda x: torch.clamp(x + 0.02*torch.randn_like(x),min=0.,max=1.)))

    data_augmentation_transform = transforms.Compose(data_augmentation_transform_array)

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
        conv_kernel_channels=CONV_KERNEL_CHANNELS,
        dropout=DROPOUT, 
        batch_normalization=BATCH_NORMALIZATION,
        beta=SWISH_BETA,
        kaiming_normal_=KAIMING_NORMAL,
        relu=RELU
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
    if SAVE_MODEL:
        torch.save(best_model, "models/model_task_2"+PATH_DETAILS+".pt")

    # Test model
    test_accuracy, predictions, labels = test_model(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy}%")

    values_str = (
        f'Split Ratio: {SPLIT_RATIO}   Batch Size: {BATCH_SIZE}   Total Epochs: {len(val_accuracies)-1}\n'
        f'Learning Rate: {LEARNING_RATE}   Momentum: {MOMENTUM}   Max Epochs: {MAX_EPOCHS}\n'    
        f'Conv Kernel Sizes: {CONV_KERNEL_SIZES}   Conv Kernel Channels: {CONV_KERNEL_CHANNELS}    Dropout: {DROPOUT}\n'
        f'Batch Normalization: {BATCH_NORMALIZATION}   Number of Models: {NUMBER_OF_MODELS}\n'
    )

    # Save plot
    save_loss_accuracy_plot(
        train_losses, 
        val_losses, 
        train_accuracies, 
        val_accuracies, 
        values_str, 
        "plots/loss_and_accuracy_task_2"+PATH_DETAILS+".png"
    )

    # Save confusion matrix
    save_confusion_matrix(
        confusion_matrix(labels, predictions),
        test_accuracy,
        "plots/confusion_matrix_task_2"+PATH_DETAILS+".png"
    )

    return test_accuracy






def task_3(SPLIT_RATIO=_SPLIT_RATIO, BATCH_SIZE=_BATCH_SIZE, LEARNING_RATE=_LEARNING_RATE, MOMENTUM=_MOMENTUM, MAX_EPOCHS=_MAX_EPOCHS,
            EPOCH_PRINT_RATIO=_EPOCH_PRINT_RATIO, EARLY_STOPPING=_EARLY_STOPPING, PATIENCE=_PATIENCE, EPOCHS_SVM=_EPOCHS_SVM, 
            KAIMING_NORMAL=_KAIMING_NORMAL, PATH_DETAILS=_PATH_3_DETAILS, SAVE_MODEL=_SAVE_MODELS, LOAD_SVM=_LOAD_SVM, SVM_C=_SVM_C,
            SVM_TYPE=_SVM_TYPE) -> tuple[float, float]:
    
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

    model = CNN_task_3_classifier(
        kaiming_normal_=KAIMING_NORMAL
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

    # We only have to save the last layer of the model, since the rest is frozen and can be loaded from torchvision.
    if SAVE_MODEL:
        torch.save(model.alexnet.classifier[6].state_dict(), "models/model_task_3"+PATH_DETAILS+".pt")

    # Test model
    test_accuracy, predictions, labels = test_model(model, test_loader, device)
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
        values_str, 
        "plots/loss_and_accuracy_task_3_classifier"+PATH_DETAILS+".png"
    )

    # Save confusion matrix
    save_confusion_matrix(
        confusion_matrix(labels, predictions),
        test_accuracy,
        "plots/confusion_matrix_task_3_classifier.png"
    )

    previous_test_accuracy : float = test_accuracy

    # SVM model
    svm_model = CNN_task_3_svm(
        train_dataloader=train_loader, 
        path="models/model_task_3_svm",
        device=device,
        epochs=EPOCHS_SVM,
        svm_type=SVM_TYPE,
        load_model=LOAD_SVM,
        C=SVM_C
    ).to(device)

    # Test accuracy
    test_accuracy, predictions, labels = test_model(svm_model, test_loader, device)
    print(f"Test accuracy: {test_accuracy}%")

    # Save confusion matrix
    save_confusion_matrix(
        confusion_matrix(labels, predictions),
        test_accuracy,
        "plots/confusion_matrix_task_3_svm"+PATH_DETAILS+".png"
    )

    svm_model.save_model("models/model_task_3_svm")

    return (previous_test_accuracy, test_accuracy)