#!/usr/bin/python3

from srcs import *
import argparse


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--model", "-m", type=str, default="CNN")
    parser.add_argument("--loadweights", "-l", type=str)
    parser.add_argument("--prediction", "-p", type=str, default="predictions.csv")
    args = parser.parse_args()
    return args


def predict_indir(args, model, device):
    dataloaders = batch_test_dataloader(args.path)
    pred = test(model, dataloaders, device)
    pred_arr =  np.array(pred).reshape(-1, 1)
    np.savetxt(args.prediction, pred_arr, fmt="%s", delimiter=";")
    print(f"[Predcitons are saved => ({args.prediction})]")


def predict_single(args, model, device):
    dataloaders = img_test_dataloader(args.path)
    pred = test(model, dataloaders, device)
    img1 = plt.imread(args.path)
    img2 = img_detect_leaf(img1)
    show_image(img1, img2, pred[0])


def main():
    try:
        args = parse_arg()
        model = select_model(args.model)
        try:
            device = use_device(model)
            load_weights(model, args.loadweights, device)
            if os.path.isdir(args.path):
                predict_indir(args, model, device)
            else:
                predict_single(args, model, device)

        except KeyboardInterrupt:
            pass

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()