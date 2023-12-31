import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets

path = "dataset/train/Bedroom/image_0002.jpg"
SPLIT_RATIO = 0.85
BATCH_SIZE = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    
    torch.manual_seed(0)
    # Anisotropic scaling and conversion to tensor:
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    full_training_data = datasets.ImageFolder(root="dataset/train", transform=transform)
    test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)

    train_size = int(SPLIT_RATIO * len(full_training_data))
    val_size = len(full_training_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_training_data, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)