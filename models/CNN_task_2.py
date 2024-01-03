import torch.nn as nn

class CNN_task_2(nn.Module):
    def __init__(self, conv_kernel_sizes : int, dropout : float, batch_normalization = True, input_height : int = 64,
                 input_width : int = 64, mean_initialization : float = 0.0, std_initialization : float = 0.01, padding : int = 1):
        
        super().__init__()

        # Hyperparameters:
        self.height = input_height
        self.width = input_width
        self.padding = padding
        self.mean_initialization = mean_initialization
        self.std_initialization = std_initialization
        self.batch_normalization = batch_normalization

        self.conv_kernel_sizes = conv_kernel_sizes
        self.dropout = dropout

        # The amount of features in the last linear layer is calculated as follows (with k = (kernel_size/2-padding)):
        # (height/2 - 3*k)/2-2*k x (width/2 - 3*k)/2-2*k x 32
        # Ideally, k should be equal to 0.
        # Some example values (64x64 input):
        # Padding = 0, kernel_size = 3: 12x12x32 = 4608
        # Padding = 1, kernel_size = 3: 16x16x32 = 8192
        # Note: all divisions are integer divisions.
        self.kernel_offset = conv_kernel_sizes//2-self.padding
        self.last_input_size = ((self.height//2-3*self.kernel_offset)//2-2*self.kernel_offset)*((self.width//2-3*self.kernel_offset)//2-2*self.kernel_offset)*32
        
        # Layers:
        self.layers_list = [
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=self.conv_kernel_sizes, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.conv_kernel_sizes, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.conv_kernel_sizes, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Flatten(),
            nn.Linear(in_features=self.last_input_size, out_features=15),
            nn.Dropout(p=self.dropout),
            nn.Softmax(dim=1)
        ]

        # Add batch normalization layers if required:
        if batch_normalization:
            self.layers_list.insert(1, nn.BatchNorm2d(8))
            self.layers_list.insert(6, nn.BatchNorm2d(16))
            self.layers_list.insert(11, nn.BatchNorm2d(32))


        self.layers = nn.Sequential(*self.layers_list)

        # Initialize weights:
        self.layers.apply(self._init_weights)


    # Function for the weight initialization:
    def _init_weights(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=self.mean_initialization, std=self.std_initialization)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)


    def forward(self, x):
        return self.layers(x)