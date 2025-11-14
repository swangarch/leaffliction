#!/usr/bin/python3

from srcs import *
import argparse


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--model", "-m", type=str, default="CNN")
    parser.add_argument("--loadweights", "-l", type=str)
    parser.add_argument("--saveweights", "-s", type=str, default="weights.pth")
    parser.add_argument("--learningrate", "-lr", type=float, default=0.001)
    parser.add_argument("--batchsize", "-bs", type=int, default=64)
    args = parser.parse_args()
    return args


def main():
    try:
        args = parse_arg()
        model = select_model(args.model)
        try:
            device = use_device(model)
            load_weights(model, args.loadweights, device)
            dataloaders = create_dataloader(args.path, args.batchsize)
            train_model(model, dataloaders, device, lr=args.learningrate)

        except KeyboardInterrupt:
            pass
        save_model(model, args.saveweights)

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()