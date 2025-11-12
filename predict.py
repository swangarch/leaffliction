#!/usr/bin/python3
import torch
from srcs.model import *
from srcs.model import *
import sys
from srcs.data_loader import *
from srcs.train_utils import *
from PIL import Image


def main():
    try:
        if len(sys.argv) != 3 and len(sys.argv) != 2:
            raise TypeError("Wrong argument number.")
        
        model = CNN()
        try:
            device = use_device()
            model.to(device)
            if len(sys.argv) == 3:
                model.load_state_dict(torch.load(sys.argv[2], map_location=device))
            dataloaders = test_dataloader(sys.argv[1], 64)
            test(model, dataloaders, device)

        except KeyboardInterrupt:
            pass

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()