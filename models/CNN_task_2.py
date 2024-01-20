import torch
import torch.nn as nn

class CNN_task_2(nn.Module):

    def __init__(self, conv_kernel_sizes : tuple[int, ...], conv_kernel_channels : tuple[int, ...], dropout : float, batch_normalization = True, 
                 input_height : int = 64, input_width : int = 64, kaiming_normal_ : bool = True, mean_initialization : float = 0.0, 
                 std_initialization : float = 0.01, beta : float = 1.0, relu : bool = False):
        
        super().__init__()

        # Hyperparameters:
        self.height = input_height
        self.width = input_width
        self.conv_kern_sizes = conv_kernel_sizes
        self.conv_kern_channels = conv_kernel_channels
        self.mean_initialization = mean_initialization
        self.std_initialization = std_initialization
        self.batch_normalization = batch_normalization

        self.kaiming_normal_ = kaiming_normal_
        self.conv_kernel_sizes = conv_kernel_sizes
        self.dropout = dropout

        self.relu = relu
        self.beta = beta

        # The amount of features in the last linear layer.
        self.last_input_size = (self.height//(2**(len(conv_kernel_sizes)-1)))*(self.width//(2**(len(conv_kernel_sizes)-1)))*self.conv_kern_channels[-1]
        
        # Layers:
        self.layers_list = []

        for i, conv_size in enumerate(self.conv_kernel_sizes):
            self.layers_list.append(nn.Conv2d(in_channels=(self.conv_kern_channels[i-1] if i != 0 else 1), out_channels=self.conv_kern_channels[i], kernel_size=conv_size, stride=1, padding=conv_size//2))
            if self.batch_normalization:
                self.layers_list.append(nn.BatchNorm2d(self.conv_kern_channels[i]))
            if relu:
                self.layers_list.append(nn.ReLU())
            else:
                self.layers_list.append(Swish(beta=self.beta))
            if self.dropout > 0.0:
                self.layers_list.append(nn.Dropout(p=self.dropout))
            if i < len(self.conv_kernel_sizes)-1:
                self.layers_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Add the last linear layer:
        self.layers_list.append(nn.Flatten())
        self.layers_list.append(nn.Linear(in_features=self.last_input_size, out_features=15))

        self.layers = nn.Sequential(*self.layers_list)

        # Initialize weights:
        self.layers.apply(self._init_weights)


    # Function for the Kaiming initialization:
    def _init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            if self.kaiming_normal_:
                if type(m) == nn.Conv2d:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.normal_(m.weight, mean=self.mean_initialization, std=self.std_initialization)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)


    def forward(self, x):
        return self.layers(x)
    
class ensemble_task_2(nn.Module):

    def __init__(self, number_of_models : int, conv_kernel_sizes : tuple[int, ...], conv_kernel_channels : tuple[int, ...], dropout : float, 
                 batch_normalization = True, input_height : int = 64, input_width : int = 64, kaiming_normal_ : bool = True, 
                 mean_initialization : float = 0.0, std_initialization : float = 0.01, beta : float = 1.0, relu : bool = False):
        super().__init__()

        self.models = nn.ModuleList([
            CNN_task_2(
                conv_kernel_sizes=conv_kernel_sizes,
                conv_kernel_channels=conv_kernel_channels,
                dropout=dropout, 
                batch_normalization=batch_normalization, 
                input_height=input_height, 
                kaiming_normal_=kaiming_normal_, 
                input_width=input_width, 
                mean_initialization=mean_initialization, 
                std_initialization=std_initialization, 
                beta=beta,
                relu=relu
            ) for _ in range(number_of_models)])

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return sum(outputs) / len(outputs)
    
# Swish activation function:
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)