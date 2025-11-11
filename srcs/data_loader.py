from torch.utils.data import DataLoader
import torch


def load_data(X, y, batch_size, shuffle):
    X_tensor = torch.tensor(X).unsqueeze(1).float()
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