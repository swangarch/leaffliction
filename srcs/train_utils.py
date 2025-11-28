import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import List
from .model import CNN, RESNET


def training(model: nn.Module, train_loader: DataLoader,
             optimizer: Optimizer, device: str) -> tuple[float, float]:
    """Perform training phase, including feedforward, backpropagation,
    and gradient descent, iterating all batches form a dataset, generate
    predictions and then calculate accuracy."""
    model.train()
    correct_count, loss_epoch = 0, 0
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)
        correct_count += pred.eq(label.view_as(pred)).sum().item()
        loss = F.cross_entropy(output, label.squeeze())
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    # number of batches in one epoch
    loss_epoch /= len(train_loader)
    # number of samples in all dataset
    correct_rate = correct_count / len(train_loader.dataset)
    return loss_epoch, correct_rate


def validation(model: nn.Module, val_loader: DataLoader,
               device: str) -> tuple[float, float]:
    """Perform validation phase, use model to do inference
    and calculate loss and accuracy."""
    model.eval()
    correct_count, loss_val = 0, 0
    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True)
            correct_count += pred.eq(label.view_as(pred)).sum().item()
            loss_val += F.cross_entropy(output, label.squeeze())
    loss_val /= len(val_loader)
    correct_rate = correct_count / len(val_loader.dataset)
    return float(loss_val), correct_rate


def test(model: nn.Module, test_loader: DataLoader,
         device: str) -> List[int]:
    """Perform test phase, use model to do inference and return
    the prediction numeric labels in a list."""
    print("[Predicting on test data...]")
    prediction = []
    model.eval()
    with torch.no_grad():
        for sample in test_loader:
            img = sample[0] if len(sample) == 2 else sample
            img = img.to(device)
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True)
            prediction.append(pred.cpu().numpy())
    out = []
    for batch in prediction:
        for p in batch:
            out.append(int(p))
    return out


def is_early_stopped(loss_epoch: float, train_loss: float,
                     counter: int) -> tuple[int, bool]:
    """Check if model loss didn't change during 3 continuous
    epochs, the training will stop to prevent overfitting."""
    if loss_epoch is not None and abs(train_loss - loss_epoch) < 0.001:
        counter += 1
    else:
        counter = 0
    if counter >= 3:
        print("[Early stopped.]")
        return counter, True
    return counter, False


def train_model(model: nn.Module, dataloaders: tuple[DataLoader, DataLoader],
                device: str, lr: float = 0.001,
                max_epoch: int = 20) -> nn.Module:
    """Perform mini training and validation phase, and collect result,
    return the model."""
    os.makedirs("visualize", exist_ok=True)
    train_loader, val_loader = dataloaders
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_epoch = None
    counter = 0
    print("[Training started]")
    records = [[], [], [], []]
    for epoch in range(max_epoch):
        val_loss, acc_val = validation(model, val_loader, device)
        t_loss, acc_t = training(model, train_loader, optimizer, device)
        counter, early_stop = is_early_stopped(loss_epoch, t_loss, counter)
        if early_stop is True:
            break
        record = [t_loss, val_loss, acc_t, acc_val]
        print(f"[Epoch] {epoch}  "
              f"[Train Loss] {record[0]:.4f}  "
              f"[Train Acc] ({(record[2] * 100):.0f}%)  "
              f"[Val Loss] {record[1]:.4f}  "
              f"[Val Acc]: ({(record[3] * 100):.0f}%)")
        append_records(records, record)
        loss_epoch = t_loss
    print("[Training done.]")
    show_records(records)
    return model


def append_records(records: list[list[float]], record: list[float]) -> None:
    """Store the training metrics."""
    loss_epoch, val_loss, acc_train, acc_val = record
    loss_train_all, loss_val_all, acc_train_all, acc_val_all = records

    loss_train_all.append(float(loss_epoch))
    loss_val_all.append(float(val_loss))
    acc_train_all.append(float(acc_train))
    acc_val_all.append(float(acc_val))


def show_records(records: list[list[float]]) -> None:
    """Visualize the training curve and accuracy curve at the
    end of training."""
    loss_train_all, loss_val_all, acc_train_all, acc_val_all = records
    plt.title("Loss curves")
    plt.plot(loss_train_all, label="Train loss")
    plt.plot(loss_val_all, label="Validation loss")
    plt.legend(loc="upper right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("visualize/Loss.jpg")
    plt.close()

    plt.title("Accuracy curves")
    plt.plot(acc_train_all, label="Train accuracy")
    plt.plot(acc_val_all, label="Validation accuracy")
    plt.legend(loc="lower right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("visualize/Accuracy.jpg")
    plt.close()


def use_device(model: nn.Module) -> str:
    """Check if CUDA is available, if available use GPU, otherwise use CPU."""
    print(f"[CUDA => ({torch.cuda.is_available()})]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return device


def select_model(name: str, num_categories: int) -> CNN | RESNET:
    """Select training model according to the name."""
    if name == "RESNET":
        print("[Using CNN Resnet]")
        model = RESNET(num_classes=num_categories)
    elif name == "CNN":
        print("[Using CNN]")
        model = CNN(out_channels=num_categories)
    else:
        raise TypeError("Unsupported model.")
    return model


def save_model(model: nn.Module, path: str) -> None:
    """Save the model weights after training."""
    torch.save(model.state_dict(), path)
    print(f"[Model saved => ({path})]")


def load_weights(model: nn.Module, weights_path: str, device: str) -> None:
    """Load weights from external weights file."""
    if weights_path is None:
        return
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[Pretrained weights => ({weights_path}) loaded]")
    except Exception as e:
        raise ValueError("""Cannot load weights,
                         please make sure weights match network""") from e
