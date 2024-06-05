from torch import nn

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
        # ... rest of your code

    def forward(self, x):
        residual = self.maxpool(self.downsample1(x))
        # print('residual shape:', residual.shape)
        # print('x shape:', x.shape)
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

        # ... rest of your code