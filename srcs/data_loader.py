import torch
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def create_dataloader(path:str="./leaves/images/", batch_size:int=64, train_ratio:float=0.8) -> None:
	img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
			mean=[0.5, 0.5, 0.5],
			std=[0.5, 0.5, 0.5])
    ])
	dataset = datasets.ImageFolder(
		root=path,
		transform=img_transform,
		allow_empty=True)
	print(dataset.classes)
	len_train = int(len(dataset) * train_ratio)
	len_val = len(dataset) - len_train
	train_set, val_set = random_split(dataset, [len_train, len_val])
	# print(train_set[9])
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
	return train_loader, val_loader


def test_dataloader(path:str="./leaves/images/", batch_size:int=64, train_ratio:float=0.8) -> None:
	img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
			mean=[0.5, 0.5, 0.5],
			std=[0.5, 0.5, 0.5])
    ])
	dataset = datasets.ImageFolder(
		root=path,
		transform=img_transform,
		allow_empty=True)
	test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return test_loader
