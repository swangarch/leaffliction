import torch
import torch.optim as optim
import argparse
import segmentation_models_pytorch as smp

from srcs import (
                    prepare_loader,
                    training_segmentation,
                    inference_segmentation,
                    use_device
                )


def parse_arg() -> argparse.Namespace:
    """Parse arguments and handle options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", default="leaves_segmentation")
    parser.add_argument("--model", "-m", type=str, default="UNET_RESNET18")
    parser.add_argument("--train", "-t", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    if not args.train:
        train_loader, val_loader, test_loader = prepare_loader("leaves_segmentation/", "leaf")
        print("Start infenrence mode")
        model_leaf = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
        model_disease = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
        try:
            model_leaf.load_state_dict(torch.load("weights/weights_leaf.pth", map_location="cpu", weights_only=True))
            model_disease.load_state_dict(torch.load("weights/weights_disease.pth", map_location="cpu", weights_only=True))
        except FileNotFoundError as e:
            print(f"Error : {e}")
        inference_segmentation(test_loader, model_leaf, model_disease)

    else:
        for task in ["leaf", "disease"]:
            print(f"Training model for taks: {task}")
            train_loader, val_loader, test_loader = prepare_loader("leaves_segmentation/", task)
            model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
            dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            device = use_device(model)
            
            training_segmentation(train_loader,
                                  val_loader,
                                  device,
                                  model, 
                                  optimizer, 
                                  dice, 
                                  f"weights/weights_{task}.pth")


if __name__ == "__main__":
    main()
