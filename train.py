#!/usr/bin/python3


import torch
from srcs.model import *
from srcs.model import *
import sys
from srcs.data_loader import *
from srcs.train_utils import *



# def visu(data):
#     train_df, datas = read_train_datas(data)
#     show_train_datas(train_df)


# def cnn_train(model, data, device):
#     train_df, datas = read_train_datas(data)
#     train_model(model, datas, device)


# def cnn_predict(model, testdata, device):
#     test_df, X_test = read_test_datas(testdata)
#     new_test_loader = load_test_datas(X_test)
#     predictions = test(model, new_test_loader, device)
#     show_result(test_df, predictions)


# def cnn_load_weights(model, weights, device):
#     model.load_state_dict(torch.load(weights, map_location=device))


def main():
    # try:
        model = CNN()
        # try:
        device = use_device()
        model.to(device)
        in_arr, out_arr = load_image()
        datas = split_datas(in_arr, out_arr)
        train_model(model, datas, device)

    #     except KeyboardInterrupt:
    #         save_model(model, "weights.pth")
    #         print("Stopped by user, weights saved => (weights.pth)")

    # except Exception as e:
    #     print("Error:", e)


if __name__ == "__main__":
    main()