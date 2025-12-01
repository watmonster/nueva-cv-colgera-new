#!/usr/bin/env python

# from example_module import a_function_from_another_module

from skimage import io  # type: ignore
from skimage import color  # type: ignore
import numpy as np
import matplotlib.pyplot as plt

# from time import time
import cv2


# Make matplotlib figures appear inline in the notebook rather than in a new window
# %matplotlib inline
plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def load(image_path):
    out = io.imread(image_path)
    out = out.astype(np.float64) / 255
    return out


def display(img):
    # Show image
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def load_my_image():
    image_path = "./my_project/column_0011.png"
    image = load(image_path)
    return image
    # display(image)


def make_grayscale(img):
    return color.rgb2gray(img).astype("float32")
    

def corner_detection(img):
    gray_img = make_grayscale(img)
    corners = np.copy(gray_img)
    cv2.cornerHarris(gray_img, 4, 3, 0.04, corners)  # noqa
    # display(corners)
    return corners
    # corners = cv2.dilate(corners, None)
    # image[corners > 0.01 * corners.max()] = [0, 0, 255]
    # cv2.imshow("corners??", gray_img)


def clamp(img):
    clamped_img = np.copy(img)
    clamped_img[clamped_img < 0] = 0
    clamped_img[clamped_img > np.max(clamped_img) / 100] = 1
    return clamped_img


# image = load_my_image()
# image_corners = corner_detection(image)
# print(image_corners[1])
