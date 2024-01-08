import torch
import torch.nn as nn

class CNN_task_2(nn.Module):

    def __init__(self, conv_kernel_sizes : tuple[int, int, int], dropout : float, batch_normalization = True, input_height : int = 64, kaiming_normal_ : bool = True,
                 input_width : int = 64, mean_initialization : float = 0.0, std_initialization : float = 0.01, padding : tuple[int, int, int] = (1, 1, 1),
                 beta : float = 1.0, cap : float = 10.0):
        
        super().__init__()

        # Hyperparameters:
        self.height = input_height
        self.width = input_width
        self.padding = padding
        self.mean_initialization = mean_initialization
        self.std_initialization = std_initialization
        self.batch_normalization = batch_normalization

        self.kaiming_normal_ = kaiming_normal_
        self.conv_kernel_sizes = conv_kernel_sizes
        self.dropout = dropout

        self.beta = beta
        self.cap = cap

        # The amount of features in the last linear layer is calculated as follows (with k_n = (kernel_size_n/2-padding_n)):
        # (height/2-k_1-2*k_2)/2-2*k_3 x (width/2-k_1-2*k_2)/2-2*k_3 x 32
        # Ideally, all k_n should be equal to 0 to get height/4 x width/4 x 32.
        # Note: all divisions are integer divisions.
        self.kernel_offset = [conv_kernel_sizes[n]//2-self.padding[n] for n in range(len(conv_kernel_sizes))]
        self.last_input_size = ((self.height//2-self.kernel_offset[0]-2*self.kernel_offset[1])//2-2*self.kernel_offset[2])*((self.width//2-self.kernel_offset[0]-2*self.kernel_offset[1])//2-2*self.kernel_offset[2])*32
        
        # Layers:
        self.layers_list = [
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=self.conv_kernel_sizes[0], stride=1, padding=self.padding[0]),
            SwishC(beta=self.beta, cap=self.cap),
            nn.Dropout(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.conv_kernel_sizes[1], stride=1, padding=self.padding[1]),
            SwishC(beta=self.beta, cap=self.cap),
            nn.Dropout(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.conv_kernel_sizes[2], stride=1, padding=self.padding[2]),
            SwishC(beta=self.beta, cap=self.cap),
            nn.Dropout(p=self.dropout),
            nn.Flatten(),
            nn.Linear(in_features=self.last_input_size, out_features=15)
        ]

        # Add batch normalization layers if required:
        if batch_normalization:
            self.layers_list.insert(1, nn.BatchNorm2d(8))
            self.layers_list.insert(6, nn.BatchNorm2d(16))
            self.layers_list.insert(11, nn.BatchNorm2d(32))


        self.layers = nn.Sequential(*self.layers_list)

        # Initialize weights:
        self.layers.apply(self._init_weights)


    # Function for the He weight initialization:
    def _init_weights(self, m):
        if type(m) == nn.Conv2d:
            if self.kaiming_normal_:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.normal_(m.weight, mean=self.mean_initialization, std=self.std_initialization)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)


    def forward(self, x):
        return self.layers(x)
    
class ensemble_task_2(nn.Module):

    def __init__(self, number_of_models : int, conv_kernel_sizes : tuple[int, int, int], dropout : float, batch_normalization = True, 
                 input_height : int = 64, kaiming_normal_ : bool = True, input_width : int = 64, 
                 mean_initialization : float = 0.0, std_initialization : float = 0.01, padding : tuple[int, int, int] = (1, 1, 1), 
                 beta : float = 1.0, cap : float = 10.0):
        super().__init__()

        self.models = nn.ModuleList([
            CNN_task_2(
                conv_kernel_sizes=conv_kernel_sizes,
                dropout=dropout, 
                batch_normalization=batch_normalization, 
                input_height=input_height, 
                kaiming_normal_=kaiming_normal_, 
                input_width=input_width, 
                mean_initialization=mean_initialization, 
                std_initialization=std_initialization, 
                padding=padding,
                beta=beta,
                cap=cap
            ) for _ in range(number_of_models)])

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return sum(outputs) / len(outputs)
    
# Capped Swish activation function:
class SwishC(nn.Module):
    def __init__(self, beta=1.0, cap=10.0):
        super().__init__()
        self.beta = beta
        self.cap = torch.tensor(cap)

    def forward(self, x):
        return torch.where(x < self.cap, x * torch.sigmoid(self.beta * x), self.cap)