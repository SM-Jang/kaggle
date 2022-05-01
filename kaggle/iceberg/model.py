import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(CNN, self).__init__()
        self.conv1 = block(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2)
        self.conv2 = block(64, 128, 3, 2)
        self.conv3 = block(128, 128, 2, 2)
        self.conv4 = block(128, 64, 2, 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, out_channels)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.reshape(x.shape[0], -1) # (n, 256)
        x = self.classifier(x)
        return x
        
class block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size, stride)
        # self.dropout = nn.Dropout(0.2)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        # x = self.dropout(x)
        x = self.bn(x)
        return x