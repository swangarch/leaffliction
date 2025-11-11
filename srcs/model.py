import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 20, 3)
		self.conv2 = nn.Conv2d(20, 32, 3)
		self.conv3 = nn.Conv2d(32, 32, 2)
		self.dropout = nn.Dropout(0.2)
		self.fc1 = nn.Linear(32 * 2 * 2, 200)
		self.fc2 = nn.Linear(200, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.conv3(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)

		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		return x



