import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_channel, output_dim):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.downsample1 = nn.Conv2d(input_channel, 8, 1)  # 1x1 conv
        self.downsample2 = nn.Conv2d(8, 16, 1)  # 1x1 conv
        self.downsample3 = nn.Conv2d(16, 32, 1)  # 1x1 conv
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 265)  # Adjust the dimension according to your input size
        self.fc2 = nn.Linear(265, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.fc =  nn.Sequential(self.fc1, self.relu, self.fc2, self.relu, self.fc3, self.relu, self.fc4)

    def forward(self, x):
        residual = self.maxpool(self.downsample1(x))
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out += residual  # Skip connection

        residual = self.maxpool(self.downsample2(out))
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out += residual  # Skip connection

        residual = self.maxpool(self.downsample3(out))
        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out += residual  # Skip connection

        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.fc(out)  # Pass it through the fully connected layer

        return out