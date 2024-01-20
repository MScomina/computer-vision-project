import torch.nn as nn

class CNN_task_1(nn.Module):
    def __init__(self, input_height : int = 64, input_width : int = 64, mean_initialization : float = 0.0, std_initialization : float = 0.01, 
                 conv_kernel_sizes : int = 3):
        super().__init__()

        # Hyperparameters:
        self.height = input_height
        self.width = input_width
        self.mean_initialization = mean_initialization
        self.std_initialization = std_initialization
        self.conv_kernel_sizes = conv_kernel_sizes

        self.last_input_size = (self.height // 4) * (self.width // 4) * 32

        # Layers:
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=self.conv_kernel_sizes, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.conv_kernel_sizes, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.conv_kernel_sizes, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=self.last_input_size, out_features=15)
        )

        # Initialize weights:
        self.layers.apply(self._init_weights)


    # Function for the weight initialization:
    def _init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=self.mean_initialization, std=self.std_initialization)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)


    def forward(self, x):
        return self.layers(x)