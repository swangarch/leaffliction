#!/usr/bin/python3

import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from numpy import ndarray as array
import cv2


def show_image(img1: array, img2: array, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(label, fontsize=24)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].imshow(img1)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].imshow(img2)
    plt.show()
    plt.close()


def load_img(path: str) -> array:
    img, img_path, filename = pcv.readimage(filename=path)
    return img


def img_show_hist(img: array) -> None:
    hist = pcv.visualize.histogram(img=img)
    pcv.print_image(hist, "./visualization/histogram.png")


# Transform-------------------
def img_detect_leaf(img: array) -> array:
    # img = pcv.gaussian_blur(img, (5,5), sigma_x=0, sigma_y=0)
    saturation_img = pcv.rgb2gray_hsv(img, "s")
    mask = pcv.threshold.binary(gray_img=saturation_img, threshold=40, object_type='light')
    masked_img = pcv.apply_mask(img=img, mask=mask, mask_color="white")
    return masked_img


def blur_img(img: array) -> array:
    blurred_img = pcv.gaussian_blur(img, (5,5), sigma_x=0, sigma_y=0)
    pcv.print_image(blurred_img, "./visualization/blurred.png")
    return blurred_img


def hue_mask(img: array) -> array:
    hue_img = pcv.rgb2gray_hsv(img, "H")
    mask_low = pcv.threshold.binary(gray_img=hue_img, threshold=35, object_type='light')
    mask_high = pcv.threshold.binary(gray_img=hue_img, threshold=85, object_type='light')
    mask = pcv.logical_and(mask_low, mask_high)
    mask = cv2.bitwise_not(mask)
    pcv.print_image(mask, "./visualization/hue_mask.png")
    return mask


def saturation_mask(img) -> array:
    saturation_img = pcv.rgb2gray_hsv(img, "s")
    mask = pcv.threshold.binary(gray_img=saturation_img, threshold=40, object_type='light')
    pcv.print_image(mask, "./visualization/saturation_mask.png")
    return mask


def analyze_img(img, mask) -> array:
    analyze_img = pcv.analyze.size(img=img, labeled_mask=mask, label="default")
    pcv.print_image(analyze_img, "./visualization/analyze.png")
    return analyze_img


def contour_img(img, mask) -> array:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    img_contour = img.copy()
    cv2.rectangle(img_contour, (x,y),(x + w, y + h), (0,0,255), 5)
    pcv.print_image(img_contour, "./visualization/contour.png")
    return img_contour


def leaf_silhouette(img) -> array:
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    pcv.print_image(binary, "./visualization/binary.png")
    return binary


def render_points(img: array, points: list, color: tuple) -> None:
    for i in range(len(points)):
        for (x, y) in points[i]:
            cv2.circle(img, (int(x), int(y)), radius=5, color=color, thickness=-1)


def extract_pseudolandmarks(img: array, mask: array) -> array:
    top, bottom, center_v = pcv.homology.y_axis_pseudolandmarks(img=img, mask=mask)
    img_p = img.copy()
    render_points(img_p, top, (0,0,255))
    render_points(img_p, bottom, (0,255,0))
    render_points(img_p, center_v, (255,0,0))
    return img_p


def plot_histogram(img: array) -> None:
    colors = ["red", "green", "blue"]
    plt.figure(figsize=(10, 5))
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=color, label=color)
    plt.title("Image Color Histogram")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Proportion of pixels (%)")
    plt.legend()
    plt.xlim([0, 256])
    plt.show()
