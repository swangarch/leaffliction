from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os, json
from typing import List, Tuple
from torch import Tensor
from PIL import Image


def create_transform() -> transforms.Compose:
    """Transform the raw image input as Tensor, normalize it and return 
    the transformation function."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])])


def create_dataset(path:str) -> datasets.ImageFolder:
    """Load image dataset from a path, the folder structure has to be 
    [category name]/[imagename], the folder name index will be used
    as classification label."""
    return datasets.ImageFolder(
        root=path,
        transform=create_transform(),
        allow_empty=True)


def create_dataloader(path:str, batch_size:int=64, train_ratio:float=0.8) -> tuple[tuple[DataLoader, DataLoader], int]:
    """Create the training dataset from an image folder, the internal 
    folder structure has to be [category name]/[imagename], split 
    the dataset to traning set and validation set. Return the training 
    data loader and validation data loader, and return the categories 
    count."""
    dataset = create_dataset(path)
    len_train = int(len(dataset) * train_ratio)
    len_val = len(dataset) - len_train
    train_set, val_set = random_split(dataset, [len_train, len_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    save_categories(dataset.classes)
    return (train_loader, val_loader), len(dataset.classes)


def batch_test_dataloader(path:str, batch_size:int=64) -> Tuple[DataLoader, datasets.ImageFolder]:
    """Create the test dataset from an image folder, the internal 
    folder structure has to be [category name]/[imagename], Return 
    the test data loader and the dataset."""
    dataset = create_dataset(path)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_loader, dataset


def img_test_dataloader(path:str) -> DataLoader[Tensor]:
    """Create the test dataset from a single image path, Return 
    the test data loader without lable."""
    img_transform = create_transform()
    img = Image.open(path).convert("RGB")
    img_tensor = img_transform(img.copy()).unsqueeze(0)
    test_loader = DataLoader(img_tensor, batch_size=1, shuffle=False)
    return test_loader


def save_categories(categories: List[str], path:str="./tmp/categories.json") -> None:
    """Save the categories in a json file, during the training phase,
    to pass the string label to prediction phase, the prediction program
    will load these labels."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    print(f"[Categories saved at (./tmp/categories.json) with {len(categories)} categories.]")


def load_categories(path:str="./tmp/categories.json") -> List[str]:
    """During the prediction the model output will be int, so it needs to 
    load string category names saved by the training program."""
    if not os.path.exists(path):
        raise RuntimeError("No categories file, cannot determine the catergories.")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)