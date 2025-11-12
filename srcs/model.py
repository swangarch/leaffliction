import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
      super(CNN, self).__init__()
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
