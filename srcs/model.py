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
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return F.relu(out)


def make_stage(in_channels, out_channels, num_blocks):
    layers = []
    # first block downsamples
    layers.append(ResidualBlock(in_channels, out_channels, downsample=True))
    # remaining blocks keep size
    for _ in range(num_blocks - 1):
        layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class RESNET(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        # input 256 → 128
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = make_stage(32, 32, num_blocks=2)     # 128x128
        self.layer2 = make_stage(32, 64, num_blocks=2)     # 64x64
        self.layer3 = make_stage(64, 128, num_blocks=2)    # 32x32
        self.layer4 = make_stage(128, 256, num_blocks=2)   # 16x16

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x