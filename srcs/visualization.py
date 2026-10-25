import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from matplotlib.colors import ListedColormap


def show_segmentaion(image: torch.Tensor, pred_leaf: torch.Tensor, pred_disease: torch.Tensor, id: int):

    fig, ax = plt.subplots(2, 3, figsize=[12, 8])

    leaf_array = pred_leaf[0].cpu().numpy()
    disease_array = pred_disease[0].cpu().numpy()
    leaf_mask = np.ma.masked_where(leaf_array == 0, leaf_array)
    disease_mask = np.ma.masked_where(disease_array == 0, disease_array)

    leaf_area = np.count_nonzero(leaf_mask)
    disease_area = np.count_nonzero(disease_mask)

    ratio = disease_area / leaf_area if leaf_area > 0 else 0

    ax[0, 0].imshow(image.permute(1, 2, 0))
    ax[0, 0].set_title("Original")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])

    ax[1, 0].imshow(pred_leaf.permute(1, 2, 0), cmap="gray")
    ax[1, 0].set_title("Mask")
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])

    ax[0, 1].imshow(image.permute(1, 2, 0))
    ax[0, 1].imshow(leaf_mask,
                 cmap=ListedColormap(["red"]),
                 alpha=0.5)
    ax[0, 1].set_title("Overlay")
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])

    rgb = image.permute(1, 2, 0).cpu().numpy()
    mask = pred_leaf.permute(1, 2, 0).cpu().numpy() == 1

    segmentation = rgb * mask

    ax[0, 2].imshow(np.concatenate([segmentation, mask], axis=-1))
    ax[0, 2].set_title("Segmentation")
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])


    ax[1, 1].imshow(image.permute(1, 2, 0))
    ax[1, 1].imshow(leaf_mask,
                cmap=ListedColormap(["green"]),
                alpha=0.5)
    ax[1, 1].imshow(disease_mask,
                 cmap=ListedColormap(["red"]),
                 alpha=0.5)
    ax[1, 1].set_title("Overlay")
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])

    
    ax[1, 2].imshow(leaf_mask,
                cmap=ListedColormap(["green"]),
                alpha=0.5, label="leaf")
    ax[1, 2].imshow(disease_mask,
                 cmap=ListedColormap(["red"]),
                 alpha=0.5, label="disease")
    ax[1, 2].set_title(f"Disease ratio {ratio * 100:.2f}%")
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].legend(handles=[
        Patch(facecolor="green", alpha=0.5, label="leaf"),
        Patch(facecolor="red", alpha=0.5, label="disease"),
    ])

    plt.savefig(f"output/{id}.png")
    plt.close()
