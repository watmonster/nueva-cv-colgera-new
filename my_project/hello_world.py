#!/usr/bin/env python

# from example_module import a_function_from_another_module

from skimage import io
from skimage import color
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


image_path = "./my_project/column_0011.png"
image = load(image_path)
display(image)


def corner_detection(img):
    gray_img = color.rgb2gray(img)
    corners = cv2.cornerHarris(gray_img, 4, 3, 0.04)
    display(corners)
    # corners = cv2.dilate(corners, None)
    # image[corners > 0.01 * corners.max()] = [0, 0, 255]
    # cv2.imshow("corners??", gray_img)


corner_detection(image)
