import torch.nn as nn

class CNN_task_1(nn.Module):
    def __init__(self, input_height : int = 64, input_width : int = 64, mean_initialization : float = 0.0, std_initialization : float = 0.01, conv_kernel_sizes : int = 3, padding : int = 0):
        super().__init__()

        # Hyperparameters:
        self.height = input_height
        self.width = input_width
        self.padding = padding
        self.mean_initialization = mean_initialization
        self.std_initialization = std_initialization
        self.conv_kernel_sizes = conv_kernel_sizes

        # The amounnt of features in the last linear layer is calculated as follows (with k = (1-padding)):
        # (height/2 - 3*k)/2-2*k x (width/2 - 3*k)/2-2*k x 32
        # Some example values (64x64 input):
        # Padding = 0: 12x12x32 = 4608
        # Padding = 1: 16x16x32 = 8192
        self.padding_offset = 1-self.padding
        self.last_input_size = ((self.height//2-3*self.padding_offset)//2-2*self.padding_offset)*((self.width//2-3*self.padding_offset)//2-2*self.padding_offset)*32

        # Layers:
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=self.conv_kernel_sizes, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.conv_kernel_sizes, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.conv_kernel_sizes, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=self.last_input_size, out_features=15)
        )

        # Initialize weights:
        self.layers.apply(self._init_weights)


    # Function for the weight initialization:
    def _init_weights(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=self.mean_initialization, std=self.std_initialization)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)


    def forward(self, x):
        return nn.functional.softmax(self.layers(x), dim=1)
        
        