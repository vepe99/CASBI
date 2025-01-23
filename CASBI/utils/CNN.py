import torch
import torch.nn as nn

"""

Utils function for the embedding network. In our case we adopt a Convolutional Neural Network (CNN) to extract the features from the image-like input, 
the chemical abundance histograms.
"""


class ConvNet_halo(nn.Module):
    """
    This class defines the CNN architecture for the embedding network. The CNN is used to extract the features from the chemical abundance histograms of entire galaxy halos.
    """   
    def __init__(self, output_dim):
        super(ConvNet_halo, self).__init__()
        # self.ones = torch.ones(10).float().to(self.device)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layer_1 = nn.Sequential(nn.Linear(32*8*8+1, 256), nn.ReLU())   
        self.fc_layer_2 = nn.Sequential(nn.Linear(256+1, 128), nn.ReLU())
        self.fc_layer_3 = nn.Sequential(nn.Linear(128+1, 64), nn.ReLU())
        self.fc_layer_4 = nn.Linear(64+1, output_dim)

    def set_device (self, device):
        self.device = device
        
    def forward(self, x):
        if len(x.shape) == 3:
            x_0 = x[:1, :, :]
            N = x[1:2, 0, 0].reshape((1, 1)).to(self.device)
        
            out = self.conv_layers(x_0.to(self.device))
            out = out.view(1, -1)
        else: 
            x_0 = x[:, :1, :, :] #the :1 keeps the channel dimension
            N =  x[:, 1:2, 0, 0].to(self.device)
            
            out = self.conv_layers(x_0.to(self.device))
            out = out.view(out.size(0), -1)
        
        
        # out = self.fc_layers(torch.cat((out, N), axis=1))
        out = self.fc_layer_1(torch.cat((out, N), axis=1))
        out = self.fc_layer_2(torch.cat((out, N), axis=1))
        out = self.fc_layer_3(torch.cat((out, N), axis=1))
        out = self.fc_layer_4(torch.cat((out, N), axis=1))
        
        return torch.cat((out, N), axis=1)
    
class ConvNet_subhalo(nn.Module):
    """
    This class defines the CNN architecture for the embedding network. The CNN is used to extract the features from the chemical abundance histograms of single dwarf galaxy subhalos.
    """   
    def __init__(self, output_dim):
        super(ConvNet_subhalo, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def set_device (self, device):
        self.device = device

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:1, :, :]
            # self.conv_layers = self.conv_layers.to(self.device)
            out = self.conv_layers(x.to(self.device))
            out = out.view(1, -1)
            self.fc_layers = self.fc_layers.to(self.device)
            out = self.fc_layers(out)
            
        else: 
            x = x[:, :1, :, :]
            # self.conv_layers = self.conv_layers.to(self.device)
            out = self.conv_layers(x.to(self.device))
            out = out.view(out.size(0), -1)
            self.fc_layers = self.fc_layers.to(self.device)
            out = self.fc_layers(out)
        return out
