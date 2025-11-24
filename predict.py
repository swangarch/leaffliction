#!/usr/bin/python3

from srcs import *
import argparse
import csv


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--model", "-m", type=str, default="CNN")
    parser.add_argument("--loadweights", "-l", type=str)
    parser.add_argument("--prediction", "-p", type=str, default="predictions.csv")
    args = parser.parse_args()
    return args


def save_prediction(args, filenames, pred_literals):
    with open(args.prediction, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["path", "prediction"])
        for name, pred in zip(filenames, pred_literals):
            writer.writerow([name, pred])


def predict_indir(args, model, device):
    dataloaders, dataset = batch_test_dataloader(args.path)
    categories = load_categories()
    pred = test(model, dataloaders, device)
    count = 0
    pred_literals = []
    filenames = []
    for i, p in enumerate(pred):
        if categories[p] == dataset.classes[dataset.samples[i][1]]:
            count += 1
        pred_literals.append(categories[p])
        filenames.append(dataset.samples[i][0])
    print(f"[Correct rate => ({100 * count / len(pred):.2f}%)  {count}/{len(pred)}]")
    save_prediction(args, filenames, pred_literals)
    print(f"[Predcitons are saved => ({args.prediction})]")


def predict_single(args, model, device):
    categories = load_categories()
    dataloaders = img_test_dataloader(args.path)
    pred = test(model, dataloaders, device)
    img1 = plt.imread(args.path)
    img2 = img_detect_leaf(img1)
    show_image(img1, img2, categories[pred[0]])


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