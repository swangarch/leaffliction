#!/usr/bin/python3

import matplotlib.pyplot as plt
from plantcv import plantcv as pcv


def img_detect_leaf(img):
	saturation_img = pcv.rgb2gray_hsv(img, "s")
	mask = pcv.threshold.binary(gray_img=saturation_img, threshold=40, object_type='light')

	masked_img = pcv.apply_mask(img=img, mask=mask, mask_color="white")
	return masked_img


def show_image(img1, img2, label):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(label, fontsize=16)
    axes[0].imshow(img1)
    axes[1].imshow(img2)
    plt.show()
    plt.close()