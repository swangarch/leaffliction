import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CNN(nn.Module):
    """A CNN model used for classification."""
    def __init__(self, in_channels: int = 3, out_channels: int = 8) -> None:
        """We use 4 convolutional layers and max pooling, with 2 fully
        connected layer, connected directly with output layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 20, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(20, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the feed forward"""
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """Classes to make a residual block."""
    def __init__(self, in_channels: int, out_channels: int,
                 downsample: bool = False) -> None:
        """The residual block used here use 2 convolutional layers, and 2 batch
        normalizations, we keep feature map dimension unchanged unless using
        down sampling."""
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               (3, 3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels,
                                  (1, 1), stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Perform the feed forward, after all layers in a residual block,
        we introduce residual connection as short cut to the next block."""
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        return F.relu(out)


def make_stage(in_channels: int, out_channels: int,
               num_blocks: int) -> nn.Sequential:
    """Function to make Residual blocks, a bottol neck structure, down sample
    in the first layer and keep dimensions in the next layers."""
    layers = []
    # first block downsamples
    layers.append(ResidualBlock(in_channels, out_channels, downsample=True))
    # remaining blocks keep size
    for _ in range(num_blocks - 1):
        layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class RESNET(nn.Module):
    """A CNN model with residual connection used for classification."""
    def __init__(self, num_classes: int = 8):
        """A structure with 4 stages, with 21 layers in total."""
        super().__init__()
        # input 256 → 128
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.layer1 = make_stage(32, 32, num_blocks=2)     # 128x128
        self.layer2 = make_stage(32, 64, num_blocks=2)     # 64x64
        self.layer3 = make_stage(64, 128, num_blocks=2)    # 32x32
        self.layer4 = make_stage(128, 256, num_blocks=2)   # 16x16
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the feed forward for the resnet."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
