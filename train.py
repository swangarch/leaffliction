#!/usr/bin/python3

from srcs import (create_dataloader,
                  select_model,
                  use_device,
                  load_weights,
                  train_model,
                  save_model,
                  split_dataset)
from Augmentation import aug_methods, balance_dataset
import argparse
import os
import zipfile


def parse_arg() -> argparse.Namespace:
    """Parse arguments and handle options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path_train")
    parser.add_argument("path_val", nargs="?")
    parser.add_argument("--model", "-m", type=str, default="CNN")
    parser.add_argument("--loadweights", "-l", type=str)
    parser.add_argument("--saveweights", "-s", type=str, default="weights.pth")
    parser.add_argument("--learningrate", "-lr", type=float, default=0.001)
    parser.add_argument("--batchsize", "-bs", type=int, default=64)
    # preprocessing of dataset
    parser.add_argument("--split", "-sp", action="store_true")
    parser.add_argument("--augmentation", "-a", action="store_true")
    parser.add_argument("--testset", "-t", action="store_true")
    # preprocessing of dataset
    args = parser.parse_args()
    return args


def save_zipfile(dataset: str, weights: str, save_path="leaffliction.zip"):
    """Save the trained weights and augmented dataset
    in a zip file."""
    with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, os.path.dirname(dataset))
                zipf.write(full_path, rel_path)
        zipf.write(weights, os.path.basename(weights))
    print(f"[Dataset and weights saved at => ({save_path})]")


def preprocess_dataset(path: str, is_augmentated: bool,
                       has_testset: bool) -> None:
    """Preprocess dataset, split in training data and validation
    data and optional test data, do augmentation only for the
    training dataset."""
    dataset = path
    train_set = os.path.join("dataset", dataset + "_train")
    val_set = os.path.join("dataset", dataset + "_val")
    split_dataset(path, "dataset", has_testset)
    if is_augmentated:
        print("[Augmenting dataset ...]")
        balance_dataset(train_set, aug_methods(), target_strategy="max")
    return train_set, val_set


def main():
    try:
        args = parse_arg()
        if args.split:
            train_set, val_set = preprocess_dataset(args.path_train,
                                                    args.augmentation,
                                                    args.testset)
            dataloaders, num_cat = create_dataloader(train_set,
                                                     val_set,
                                                     args.batchsize)
        else:
            if args.path_val is None:
                raise TypeError("Validation dataset is not provided.")
            dataloaders, num_cat = create_dataloader(args.path_train,
                                                     args.path_val,
                                                     args.batchsize)
        model = select_model(args.model, num_cat)
        try:
            device = use_device(model)
            load_weights(model, args.loadweights, device)
            train_model(model, dataloaders, device, lr=args.learningrate)
        except KeyboardInterrupt:
            pass
        save_model(model, args.saveweights)
        save_zipfile("dataset", "weights.pth")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
