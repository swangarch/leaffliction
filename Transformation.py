#!/usr/bin/python3

import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import cv2
import os
import sys


def load_img(path):
    img, img_path, filename = pcv.readimage(filename=path)
    return img


def img_show_hist(img):
    hist = pcv.visualize.histogram(img=img)
    pcv.print_image(hist, "./visualization/histogram.png")


def img_show_gaussian_blur(img):

    blurred_img = pcv.gaussian_blur(img, (5,5), sigma_x=0, sigma_y=0, roi=None)
    pcv.print_image(blurred_img, "./visualization/blurred.png")


def img_detect_leaf(img):

    blurred_img = pcv.gaussian_blur(img, (5,5), sigma_x=0, sigma_y=0, roi=None)

    # blurred_img = pcv.gaussian_blur(img, (5,5), sigma_x=0, sigma_y=0, roi=None)
    pcv.print_image(blurred_img, "./visualization/blurred2.png")

    hue_img = pcv.rgb2gray_hsv(img, "H")
    mask = pcv.threshold.binary(gray_img=hue_img, threshold=30, object_type='light')
    pcv.print_image(mask, "./visualization/hue_mask.png")

    saturation_img = pcv.rgb2gray_hsv(blurred_img, "s")
    mask = pcv.threshold.binary(gray_img=saturation_img, threshold=40, object_type='light')
    pcv.print_image(mask, "./visualization/saturation_mask.png")

    # bin_gauss1 = pcv.threshold.gaussian(gray_img=saturation_img, ksize=250, offset=15,
    #                                 object_type='dark')
    # pcv.print_image(bin_gauss1, "./visualization/binary_gaussian_mask.png")

    masked_img = pcv.apply_mask(img=img, mask=mask, mask_color="white")
    pcv.print_image(masked_img, "./visualization/leaf.png")

    analyze_img = pcv.analyze.size(img=img, labeled_mask=mask, label="default")
    pcv.print_image(analyze_img, "./visualization/analyze.png")

    # roi_img = pcv.roi.from_binary_image(img=img, bin_img=mask)
    # pcv.print_image(img, "./visualization/roi.png")

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    roi_contour = contour
    roi_hierarchy = hierarchy

    x, y, w, h = cv2.boundingRect(contour)
    bounding_roi = pcv.roi.rectangle(img=mask, x=x, y=y, h=h, w=w)
    img_contour = img.copy()
    cv2.rectangle(img_contour, (x,y),(x + w, y + h), (255,0,0), 5)

    pcv.print_image(img_contour, "./visualization/contour.png")


# import cv2

def leaf_silhouette(img):
    # 1. Gaussian blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # 3. Otsu threshold to create a black/white mask
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    pcv.print_image(binary, "./visualization/binary.png")

    return binary 

# def img_show_shape(img):
# 	mask_dilated = pcv.dilate(gray_img=img, ksize=3, i=1)
# 	mask_fill = pcv.fill(bin_img=mask_dilated, size=30)
# 	mask_fill = pcv.fill_holes(bin_img=mask_fill)
# 	plt.figure(figsize=(5, 5))
# 	img = pcv.analyze.size(img=img, labeled_mask=mask_fill, label="default")
# 	pcv.print_image(img, "shape.png")

def transform(img):
    
    img_show_hist(img)
    img_show_gaussian_blur(img)
    img_detect_leaf(img)
    


def main():
    # try:
        if len(sys.argv) != 2:
            raise TypeError("Wrong number of arguments")

        os.makedirs("./visualization", exist_ok=True)
        img = load_img(sys.argv[1])
        transform(img)

        img2 = cv2.imread(sys.argv[1])
        leaf_silhouette(img2)

    # except Exception as e:
    # 	print("Error:", e)
        

if __name__ == "__main__":
    main()