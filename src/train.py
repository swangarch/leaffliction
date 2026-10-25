import os
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .visualization import show_segmentaion


def train(train_loader: DataLoader, device: str | torch.device,
            model: nn.Module, optimizer: Optimizer, 
            dice: nn.Module, save_path: str) -> None:
    model = model.to(device)
    num_batch = len(train_loader)
    losses = []
    for epoch in range(10):
        loss_collect = 0
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            model.train()

            optimizer.zero_grad()
            predict = model(image)
            loss = dice(predict, label)
            # loss = bce(predict, label) + dice(predict, label)
            loss_collect += loss.item()
            loss.backward()
            optimizer.step()
        loss_collect = loss_collect / num_batch
        losses.append(losses)
        print(f"Epoch: {epoch} | train loss {loss_collect:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


def inference(test_loader: DataLoader, model_leaf: nn.Module, model_disease: nn.Module) -> None:
    model_leaf.eval()
    model_disease.eval()
    os.makedirs("output", exist_ok=True)
    id = 0
    with torch.no_grad():
        for image, mask in test_loader:
            image = image.to("cpu")
            mask = mask.to("cpu")
            pred_leaf = (model_leaf(image).sigmoid() > 0.5).float()
            pred_disease = (model_disease(image).sigmoid() > 0.5).float()
            for idx_in_batch in range(len(pred_leaf)):
                show_segmentaion(image[idx_in_batch], pred_leaf[idx_in_batch],
                                 pred_disease[idx_in_batch], id)
                print(f"ID: {id} is processed")
                id += 1
