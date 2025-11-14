#!/usr/bin/python3

import torch
import sys
from srcs import *


def main():
    try:
        if len(sys.argv) != 2:
            raise TypeError("Wrong argument number.")
        
        model = CNN()
        try:
            device = use_device()
            model.to(device)
            dataloaders = create_dataloader(sys.argv[1], 64)
            train_model(model, dataloaders, device)

        except KeyboardInterrupt:
            save_model(model, "weights.pth")
            print("Stopped by user, weights saved => (weights.pth)")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()