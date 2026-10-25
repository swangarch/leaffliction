#!/usr/bin/python3

from srcs import (select_model,
                  use_device,
                  load_weights,
                  batch_test_dataloader,
                  test,
                  img_test_dataloader,
                  img_test_dataloader_from_numpy,
                  img_detect_leaf,
                  show_image,
                  load_categories
                  )
import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from torch import nn


def parse_arg() -> argparse.Namespace:
    """Parse arguments and handle options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--model", "-m", type=str, default="CNN")
    parser.add_argument("--loadweights", "-l", type=str)
    parser.add_argument("--prediction", "-p", type=str,
                        default="predictions.csv")
    parser.add_argument("--loadcategories", "-lc", type=str,
                        default="categories/categories.json")
    args = parser.parse_args()
    return args


def save_prediction(args: argparse.Namespace,
                    filenames: List[str],
                    pred_literals: List[str]) -> None:
    """Save batch predictions in a CSV file. The predictions and the
    original filenames have to be provided."""
    with open(args.prediction, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["path", "prediction"])
        for name, pred in zip(filenames, pred_literals):
            writer.writerow([name, pred])


def predict_indir(args: argparse.Namespace, model: nn.Module,
                  device: str, categories: List[str]) -> None:
    """Generate predictions for images inside a directory that follows
    the same structure as the training data. Then compare the predicted
    labels with the true labels to calculate accuracy, and finally save
    the predictions to a CSV file."""
    dataloaders, dataset = batch_test_dataloader(args.path)
    pred = test(model, dataloaders, device)
    count = 0
    pred_literals = []
    filenames = []
    for i, p in enumerate(pred):
        if categories[p] == dataset.classes[dataset.samples[i][1]]:
            count += 1
        pred_literals.append(categories[p])
        filenames.append(dataset.samples[i][0])
    correct_rate = 100 * count / len(pred)
    print(f"[Correct rate => ({correct_rate:.2f}%)  {count}/{len(pred)}]")
    save_prediction(args, filenames, pred_literals)
    print(f"[Predictions are saved => ({args.prediction})]")


def predict_single(args: argparse.Namespace, model: nn.Module,
                   device: str, categories: List[str]) -> None:
    """Generate prediction on 1 single image, and visualize it along
    with its label."""
    dataloaders = img_test_dataloader(args.path)
    pred = test(model, dataloaders, device)
    img1 = plt.imread(args.path)
    img2 = img_detect_leaf(img1)
    show_image(img1, img2, categories[pred[0]])
    return {
        "catID": pred[0],
        "category": categories[pred[0]],
        "success": True
    }


def predict_single_server(arr: np.array,
                          model: nn.Module,
                          model_leaf: nn.Module,
                          model_disease: nn.Module,
                          device: str,
                          categories: List[str]) -> dict:
    """Return the category and original-size integer leaf/disease masks."""
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    dataloaders = img_test_dataloader_from_numpy(arr)
    pred = test(model, dataloaders, device)
    image = Image.fromarray(arr).convert("RGB")
    tensor = to_tensor(image.resize((256, 256))).unsqueeze(0).to(device)
    masks = []
    with torch.no_grad():
        for segmentation_model in (model_leaf, model_disease):
            segmentation_model.to(device)
            segmentation_model.eval()
            mask = (segmentation_model(tensor).sigmoid() > 0.5)
            mask = mask[0, 0].to(torch.uint8).cpu().numpy() * 255
            mask = Image.fromarray(mask).resize(
                image.size, Image.Resampling.NEAREST)
            masks.append(np.asarray(mask, dtype=np.uint8).tolist())
    leaf_pixels = np.asarray(masks[0], dtype=np.uint8) > 0
    disease_pixels = np.asarray(masks[1], dtype=np.uint8) > 0
    leaf_area = int(np.count_nonzero(leaf_pixels))
    disease_area = int(np.count_nonzero(leaf_pixels & disease_pixels))
    disease_ratio = 100.0 * disease_area / leaf_area if leaf_area else None
    return {
        "catID": pred[0],
        "category": categories[pred[0]],
        "success": True,
        "leaf_mask": masks[0],
        "disease_mask": masks[1],
        "disease_ratio": disease_ratio
    }


def main():
    try:
        args = parse_arg()
        categories = load_categories(args.loadcategories)
        model = select_model(args.model, num_categories=len(categories))
        try:
            device = use_device(model)
            load_weights(model, args.loadweights, device)
            if os.path.isdir(args.path):
                predict_indir(args, model, device, categories)
            else:
                predict_single(args, model, device, categories)
        except KeyboardInterrupt:
            pass
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
