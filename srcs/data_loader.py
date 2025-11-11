import torch
from torch.utils.data import DataLoader


def load_data(X, y, batch_size, shuffle):
    X_tensor = torch.tensor(X).float()
    tensor_dataset = None
    loader = None
    if y is not None:
        y_tensor = torch.tensor(y).long()
        tensor_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    else:
        tensor_dataset = torch.utils.data.TensorDataset(X_tensor)
    
    if batch_size is not None:
        loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        loader = DataLoader(tensor_dataset, shuffle=shuffle)
    return loader


def load_datas(X_train, X_val, y_train, y_val):
    train_loader = load_data(X_train, y_train, 256, True)
    val_loader = load_data(X_val, y_val, 256, True)
    return train_loader, val_loader


def load_test_datas(X_test):
    return load_data(X_test, None, None, False)


import os
import matplotlib.pyplot as plt
import numpy as np


def load_image():
    root = "leaves/images/"
    list_train = os.listdir(root) # all the label
    labels = dict()
    for i, folder in enumerate(list_train): # convert folder name to label
        labels[folder] = i
    inputs = []
    outputs = []
    img_count = 0
    for train_folder in list_train:
        imgs = os.listdir(root + train_folder)
        label = labels[train_folder]
        for img in imgs:
            img_count += 1
            img_path = root + train_folder + "/" + img
            img_arr = (plt.imread(img_path) / 255.0).astype(np.float32)
            inputs.append(img_arr.reshape(1, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))
            outputs.append(label)
            if img_count > 5000:
                break
        if img_count > 5000:
            break
    in_arr = np.vstack(inputs)
    out_arr = np.vstack(outputs).flatten()
    in_arr = np.transpose(in_arr, (0, 3, 1, 2))
    print("LOADED DATA", len(out_arr))
    print("[INPUT] =>", in_arr.shape)
    print("[OUTPUT] =>", out_arr.shape)
    return in_arr, out_arr


def split_datas(X_train, y_train):
    l = len(X_train)
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    train_size = int(0.8 * l)
    X_train, X_val, y_train, y_val = X_train[0:train_size], X_train[train_size:], y_train[0:train_size], y_train[train_size:]
    return (X_train, X_val, y_train, y_val)