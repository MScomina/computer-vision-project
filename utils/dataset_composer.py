import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def generate_dataloaders(path : str, transforms : tuple[transforms.Compose, transforms.Compose, transforms.Compose] | transforms.Compose,
                         split_ratio : float = 0.85, batch_size : int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    if not isinstance(transforms, tuple):
        # If only one transform is passed, use it for all datasets without loading them multiple times
        full_training_data = datasets.ImageFolder(root=path+"/train", transform=transforms)
        test_dataset = datasets.ImageFolder(root=path+"/test", transform=transforms)

        train_size = int(split_ratio * len(full_training_data))
        val_size = len(full_training_data) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_training_data, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return (train_loader, val_loader, test_loader)

    # Load datasets
    full_training_data = datasets.ImageFolder(root=path+"/train")
    test_dataset = datasets.ImageFolder(root=path+"/test", transform=transforms[2])

    # Split training data into training and validation sets
    # Note: i couldn't rely on torch.utils.data.random_split() because it doesn't let you specify different transforms for each dataset.
    train_size = int(split_ratio * len(full_training_data))
    indices = torch.randperm(len(full_training_data)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create new datasets with the desired transforms
    train_dataset = datasets.ImageFolder(root=path+"/train", transform=transforms[0])
    val_dataset = datasets.ImageFolder(root=path+"/test", transform=transforms[1])

    # Use the indices to get the corresponding data
    train_dataset.samples = [train_dataset.samples[i] for i in train_indices]
    train_dataset.targets = [train_dataset.targets[i] for i in train_indices]

    val_dataset.samples = [val_dataset.samples[i] for i in val_indices]
    val_dataset.targets = [val_dataset.targets[i] for i in val_indices]

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return (train_loader, val_loader, test_loader)