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

    def __init__(self, mean_initialization : float = 0.0, std_initialization : float = 0.01, kaiming_normal_ : bool = True):
        super().__init__()


        self.mean_initialization = mean_initialization
        self.std_initialization = std_initialization
        self.kaiming_normal_ = kaiming_normal_

        self.alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.alexnet.classifier[6] = nn.Linear(4096, 15)
        # Freeze all layers except the last one:
        for param in self.alexnet.parameters():
            param.requires_grad = False
        self.alexnet.classifier[6].apply(self._init_weights)
        self.alexnet.classifier[6].requires_grad = True
        self.alexnet.classifier[6].weight.requires_grad = True
        self.alexnet.classifier[6].bias.requires_grad = True

    # Function for the He weight initialization:
    def _init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            if self.kaiming_normal_:
                if type(m) == nn.Conv2d:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
            else:
                nn.init.normal_(m.weight, mean=self.mean_initialization, std=self.std_initialization)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        return self.alexnet(x)
    
class CNN_task_3_svm(nn.Module):

    def __init__(self, train_dataloader : DataLoader, path : str, C : float = 1.0, svm_type : str = "linear",
                gamma : float = 'scale', epochs : int = 1, device : torch.device = torch.device('cpu'), load_model : bool = True):
        super().__init__()

        self.device = device
        self.features_layer = models.alexnet(weights=AlexNet_Weights.DEFAULT).features.to(self.device)
        for param in self.features_layer.parameters():
            param.requires_grad = False

        self.svm = svm.SVC(kernel=svm_type, C=C, gamma=gamma)
        if os.path.exists(path + '.pkl') and load_model:
            self.load_model(path)
        else:
            self.fit_svm(train_dataloader, epochs)

    def forward(self, x):
        features = self.features_layer(x.to(self.device))
        features = features.view(x.size(0), -1)
        features_np = features.detach().cpu().numpy()
        predictions = self.svm.predict(features_np)
        probabilities = np.zeros((predictions.shape[0], self.svm.classes_.shape[0]))
        probabilities[np.arange(predictions.shape[0]), predictions] = 1
        return torch.from_numpy(probabilities).to(self.device)
    
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