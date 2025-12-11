#!/usr/bin/python3

from srcs import (select_model,
                  use_device,
                  load_weights,
                  batch_test_dataloader,
                  test,
                  img_test_dataloader,
                  img_detect_leaf,
                  show_image,
                  load_categories
                  )
import argparse
import csv
import os
import matplotlib.pyplot as plt
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


def main():
    try:
        args = parse_arg()
        categories = load_categories()
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
