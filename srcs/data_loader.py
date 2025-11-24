from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os, json


def create_dataloader(path:str, batch_size:int=64, train_ratio:float=0.8) -> None:
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
	len_train = int(len(dataset) * train_ratio)
	len_val = len(dataset) - len_train
	train_set, val_set = random_split(dataset, [len_train, len_val])
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
	save_categories(dataset.classes)
	return train_loader, val_loader


def batch_test_dataloader(path:str, batch_size:int=64, train_ratio:float=0.8) -> None:
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
	test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	return test_loader, dataset


def img_test_dataloader(path:str, batch_size=64, train_ratio:float=0):
	img_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.5, 0.5, 0.5],
			std=[0.5, 0.5, 0.5])
	])
	img = plt.imread(path)
	img_tensor = img_transform(img.copy())
	img_tensor = img_tensor.unsqueeze(0)
	test_loader = DataLoader(img_tensor, batch_size=batch_size, shuffle=True)
	return test_loader


def save_categories(categories, path="./tmp/categories.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    print("[Categories saved at (./tmp/categories.json)]")


def load_categories(path="./tmp/categories.json"):
    if not os.path.exists(path):
        raise RuntimeError("No categories file, cannot determine the catergories.")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)