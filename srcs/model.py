import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 20, (3,3), padding=1)
            self.conv2 = nn.Conv2d(20, 32, (3,3), padding=1)
            self.conv3 = nn.Conv2d(32, 64, (3,3), padding=1)
            self.conv4 = nn.Conv2d(64, 128, (3,3), padding=1)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(128 * 16 * 16, 256)
            self.fc2 = nn.Linear(256, 8)


        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2,2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2,2))
            x = self.conv3(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2,2))
            x = self.conv4(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2,2))

            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x



class ResidualBlock(nn.Module):
    def  __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or downsample:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.skip = nn.Identity()


    def forward(self, x):
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        out = F.relu(x)
        return out


class RESNET(nn.Module):
        def __init__(self, num_classes=8):
            super().__init__()
            self.layer1 = ResidualBlock(3, 32, downsample=True)
            self.layer2 = ResidualBlock(32, 64, downsample=True)
            self.layer3 = ResidualBlock(64, 128, downsample=True)
            self.layer4 = ResidualBlock(128, 128, downsample=True)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(128, num_classes)


        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.global_pool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            return x