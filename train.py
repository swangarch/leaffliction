#!/usr/bin/python3

import torch
import sys
from srcs import *
import argparse


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("imgdir")
    parser.add_argument("--model", "-m", type=str, default="CNN")
    parser.add_argument("--weights", "-w", type=str, default="weights.pth")
    parser.add_argument("--saveweights", "-s", type=str, default="weights_out.pth")
    args = parser.parse_args()
    return args


def select_model(name):
    if name == "RESNET":
        print("[Using Resnet]")
        model = RESNET()
    elif name == "CNN":
        print("[Using CNN]")
        model = CNN()
    else:
        raise TypeError("Unsupported model.")
    return model


def main():
    try:
        args = parse_arg()
        model = select_model(args.model)
        try:
            device = use_device()
            model.to(device)
            dataloaders = create_dataloader(sys.argv[1], 64)
            train_model(model, dataloaders, device, args.saveweights)

        except KeyboardInterrupt:
            savepath = args.saveweights
            save_model(model, savepath)
            print(f"Stopped by user, weights saved => ({savepath})")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()