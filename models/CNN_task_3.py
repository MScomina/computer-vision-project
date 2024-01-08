import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from sklearn import svm
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.alexnet import AlexNet_Weights

class CNN_task_3_classifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.alexnet.classifier[6] = nn.Linear(4096, 15)
        # Freeze all layers except the last one:
        for param in self.alexnet.parameters():
            param.requires_grad = False
        self.alexnet.classifier[6].requires_grad = True
        self.alexnet.classifier[6].weight.requires_grad = True
        self.alexnet.classifier[6].bias.requires_grad = True
        
    def forward(self, x):
        return self.alexnet(x)
    
class CNN_task_3_svm(nn.Module):

    def __init__(self, train_dataloader : DataLoader, path : str, linear_svm : bool = True, C : float = 1.0,
                gamma : float = 'scale', epochs : int = 1, device : torch.device = torch.device('cpu')):
        super().__init__()

        self.device = device
        self.features_layer = models.alexnet(weights=AlexNet_Weights.DEFAULT).features.to(self.device)
        for param in self.features_layer.parameters():
            param.requires_grad = False

        kernel_type = 'linear' if linear_svm else 'rbf'
        self.svm = svm.SVC(kernel=kernel_type, probability=True, C=C, gamma=gamma)
        if os.path.exists(path + '.pkl'):
            self.load_model(path)
        else:
            self.fit_svm(train_dataloader, epochs)

    def forward(self, x):
        features = self.features_layer(x.to(self.device))
        features = features.view(x.size(0), -1)
        features_np = features.detach().cpu().numpy()
        predictions = self.svm.predict_proba(features_np)
        return torch.from_numpy(predictions).to(self.device)
    
    def np_features(self, x):
        features = self.features_layer(x.to(self.device))
        features = features.view(x.size(0), -1)
        return features.detach().cpu().numpy()
    
    def fit_svm(self, dataloader, epochs):
        features = []
        labels = []
        for _ in range(epochs):
            for x, y in iter(dataloader):
                x = x.to(self.device)
                features.append(self.np_features(x))
                labels.append(y.detach().numpy())
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        self.svm.fit(features, labels)

    def save_model(self, path):
        with open(f'{path}.pkl', 'wb') as f:
            pickle.dump(self.svm, f)

    def load_model(self, path):
        with open(f'{path}.pkl', 'rb') as f:
            self.svm = pickle.load(f)