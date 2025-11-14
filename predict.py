#!/usr/bin/python3

import torch
import sys
from srcs import *


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
            
            path = sys.argv[1]
            if os.path.isdir(path):
                dataloaders = batch_test_dataloader(sys.argv[1])
                pred = test(model, dataloaders, device)
                pred_arr =  np.array(pred).reshape(-1, 1)
                np.savetxt("prediction.csv", pred_arr, fmt="%s", delimiter=";")
                print("[Predcitons are saved => (predictions.csv)]")
            else:
                dataloaders = img_test_dataloader(sys.argv[1])
                pred = test(model, dataloaders, device)
                img1 = plt.imread(path)
                img2 = img_detect_leaf(img1)
                show_image(img1, img2, pred[0])

        except KeyboardInterrupt:
            pass

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()