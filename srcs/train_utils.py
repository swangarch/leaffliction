import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from .model import CNN, RESNET


def training(model, train_loader, optimizer, device):
    model.train()
    correct_count, loss_epoch = 0, 0
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img) # inference with model
        pred = output.argmax(dim=1, keepdim=True)
        correct_count += pred.eq(label.view_as(pred)).sum().item()

        loss = F.cross_entropy(output, label.squeeze())
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    loss_epoch /= len(train_loader) # number of batches in one epoch
    correct_rate = correct_count / len(train_loader.dataset) # number of samples in all dataset
    return loss_epoch, correct_rate


def validation(model, val_loader, device):
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
    return loss_val, correct_rate


def test(model, test_loader, device):
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


def is_early_stopped(loss_epoch, train_loss, counter):
    if loss_epoch is not None and abs(train_loss - loss_epoch) < 0.001:
        counter += 1
    else:
        counter = 0
    if counter >= 3:
        print("[Early stopped.]")
        return counter, True
    return counter, False


def train_model(model, dataloaders, device, lr=0.001):
    os.makedirs("visualize", exist_ok=True)
    train_loader, val_loader = dataloaders
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_epoch = None
    counter = 0
    print("[Training started]")
    records = [[], [], [], []]
    for epoch in range(20):
        val_loss, acc_val = validation(model, val_loader, device)
        train_loss, acc_train = training(model, train_loader, optimizer, device)

        counter, early_stop = is_early_stopped(loss_epoch, train_loss, counter)
        if early_stop == True:
            break
        record = [train_loss, val_loss, acc_train, acc_val]
        print(f"[Epoch] {epoch}  [Train Loss] {record[0]:.4f}  [Train Acc] ({(record[2] * 100):.0f}%)  [Val Loss] {record[1]:.4f}  [Val Acc]: ({(record[3] * 100):.0f}%)")
        append_records(records, record)
        loss_epoch = train_loss
    print("[Training done.]")
    show_records(records)
    return model


def append_records(records, record):
    loss_epoch, val_loss, acc_train, acc_val = record
    loss_train_all, loss_val_all, acc_train_all, acc_val_all = records

    loss_train_all.append(float(loss_epoch))
    loss_val_all.append(float(val_loss))
    acc_train_all.append(float(acc_train))
    acc_val_all.append(float(acc_val))


def show_records(records):
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


def use_device(model):
    print(f"[CUDA => ({torch.cuda.is_available()})]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return device


def select_model(name):
    if name == "RESNET":
        print("[Using CNN Resnet]")
        model = RESNET()
    elif name == "CNN":
        print("[Using CNN]")
        model = CNN()
    else:
        raise TypeError("Unsupported model.")
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"[Model saved => ({path})]")


def load_weights(model, weights_path:str, device):
    if weights_path is not None:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"[Pretrained weights => ({weights_path}) loaded]")
        except Exception as e:
            raise ValueError("Cannot load weights, please make sure weights match network")
