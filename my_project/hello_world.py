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
    cv2.cornerHarris(gray_img, 5, 5, 0.06, corners)  # noqa
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


def get_corners_and_edges(img):
    corners = corner_detection(img)
    clamped_corners = clamp(corners)
    clamped_edges = clamp(corners * (-1))
    return clamped_corners, clamped_edges


def colorize_corners_and_edges(img):
    corners, edges = get_corners_and_edges(img)
    # Create a 3-channel image with zeros
    green_blue_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)

    # Put the grayscale data into the green channel (channel 1)
    green_blue_image[:, :, 1] = cv2.dilate(corners, None, iterations=5)  # type: ignore

    green_blue_image[:, :, 2] = edges

    return green_blue_image


image = load_my_image()
image_features = colorize_corners_and_edges(image)
display(image)
display(image_features)
display(image + image_features)
