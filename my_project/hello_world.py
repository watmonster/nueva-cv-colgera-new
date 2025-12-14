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


def make_grayscale(img):
    return color.rgb2gray(img).astype("float32")


def rc_to_xy(coords_list):
    return (int(round(coords_list[1])), int(round(coords_list[0])))


def corner_detection(img, blocksize=3, ksize=5, k=0.04):
    gray_img = make_grayscale(img)
    corners = np.copy(gray_img)
    cv2.cornerHarris(gray_img, blocksize, ksize, k, corners)  # noqa
    return corners


def clamp(img):
    clamped_img = np.copy(img)
    clamped_img[clamped_img < np.max(clamped_img) / 20] = 0
    clamped_img[clamped_img >= np.max(clamped_img) / 20] = 1
    return clamped_img


def get_corners_and_edges(img, blocksize=3, ksize=5, k=0.04):
    corners = corner_detection(img, blocksize, ksize, k)
    clamped_corners = clamp(corners)
    clamped_edges = clamp(corners * (-1))
    return clamped_corners, clamped_edges


def colorize_corners_and_edges(img, blocksize=3, ksize=5, k=0.04):
    corners, edges = get_corners_and_edges(img, blocksize, ksize, k)
    # Create a 3-channel image with zeros
    green_blue_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)

    # Put the grayscale data into the green channel (channel 1)
    green_blue_image[:, :, 1] = cv2.dilate(corners, None, iterations=5)  # type: ignore

    green_blue_image[:, :, 2] = edges

    return green_blue_image


def load_real_image():
    return load("./my_project/real_shapes.png")


def load_test_image():
    return load("./my_project/test_shapes.png")[:, :, 0:3]


quads = {1: [-1, 1], 2: [-1, -1], 3: [1, -1], 4: [1, 1]}


def avg_pix_brightness(img, pix_loc, quad_num, radius):
    gray_img = make_grayscale(img)
    gray_clamped_img = clamp(gray_img)
    # display(gray_clamped_img)
    total_brightness = 0
    pix_loc_row = int(round(pix_loc[0]))
    pix_loc_col = int(round(pix_loc[1]))
    row_mult = quads[quad_num][0]
    col_mult = quads[quad_num][1]
    for row in range(radius):
        for col in range(radius):
            if row != 0 and col != 0:
                check_row = int(pix_loc_row + row * row_mult)
                check_col = int(pix_loc_col + col * col_mult)
                current_brightness = gray_clamped_img[check_row, check_col]
                total_brightness += current_brightness
    return total_brightness / (radius**2)


def find_corner_orientation(img, corner_loc, radius=10):
    quad_1_brightness = avg_pix_brightness(img, corner_loc, 1, radius)
    quad_2_brightness = avg_pix_brightness(img, corner_loc, 2, radius)
    quad_3_brightness = avg_pix_brightness(img, corner_loc, 3, radius)
    quad_4_brightness = avg_pix_brightness(img, corner_loc, 4, radius)
    brightness_list = [quad_1_brightness, quad_2_brightness, quad_3_brightness, quad_4_brightness]
    max_brightness = max(brightness_list)
    max_quad = brightness_list.index(max_brightness) + 1
    return max_quad


def find_centroids(harris_result):
    step1 = cv2.dilate(harris_result, None, iterations=5)  # type: ignore
    _, step2 = cv2.threshold(step1, 0.01 * np.max(step1), 255, 0)
    step3 = np.uint8(step2)
    centroids = cv2.connectedComponentsWithStats(step3)[3]  # type: ignore
    revised_centroids = centroids[1:]  # assume the centroid of the background is first?? (we think)
    flipped_centroids = [[float(centroid[1]), float(centroid[0])] for centroid in revised_centroids]
    return flipped_centroids


def centroids_and_orientations(img, centroids, radius=5):
    centroids_and_quads = []
    for centroid in centroids:
        # print(centroid)
        # print(find_corner_orientation(img, centroid, radius))
        # print(centroid.append(find_corner_orientation(img, centroid, radius)))
        centroids_and_quads.append([centroid[0], centroid[1], find_corner_orientation(img, centroid, radius)])
    return centroids_and_quads


def make_circles(img, pixel_locs):
    circles_img = np.copy(img)
    for pix in pixel_locs:
        tuple_pix = rc_to_xy(pix)
        circles_img = cv2.circle(circles_img, tuple_pix, 20, (255, 0, 0), 3)
    return circles_img


def make_orientation_marks(img, pixel_locs_orientations):
    orientations_img = np.copy(img)
    for pix in pixel_locs_orientations:
        tuple_pix = rc_to_xy(pix)
        line_end = (tuple_pix[0] + quads[pix[2]][1] * 30, tuple_pix[1] + quads[pix[2]][0] * 30)
        orientations_img = cv2.line(orientations_img, tuple_pix, line_end, (255, 0, 0), 5)
    return orientations_img


def find_one_rectangle(oriented_centroids, first_corner):  # noqa
    # print(oriented_centroids)
    rectangle = []
    # first_corner = []
    # for corner in oriented_centroids:
    #     if corner[2] == 4:
    #         first_corner = corner
    #         break
    # if not first_corner:
    #     print("couldn't find rectangle: quad 4 missing")
    #     return False
    rectangle.append(first_corner)
    print(first_corner)
    closest_quad_3 = oriented_centroids[0]
    for corner in oriented_centroids:
        if corner[2] == 3 and closest_quad_3[2] != 3:
            if corner[1] > first_corner[1]:
                if corner[0] + 2 >= first_corner[0] and corner[0] - 2 <= first_corner[0]:
                    closest_quad_3 = corner
        elif corner[2] == 3:
            if corner[1] > first_corner[1]:
                if corner[0] + 2 >= first_corner[0] and corner[0] - 2 <= first_corner[0]:
                    if corner[0] < closest_quad_3[0]:
                        closest_quad_3 = corner
    if closest_quad_3[2] != 3:
        print("couldn't find rectangle: quad 3 missing")
        return rectangle
    rectangle.append(closest_quad_3)
    closest_quad_1 = oriented_centroids[0]
    for corner in oriented_centroids:
        if corner[2] == 1 and closest_quad_1[2] != 1:
            if corner[0] > first_corner[0]:
                if corner[1] + 2 >= first_corner[1] and corner[1] - 2 <= first_corner[1]:
                    closest_quad_1 = corner
        elif corner[2] == 1:
            if corner[0] > first_corner[0]:
                if corner[1] + 2 >= first_corner[1] and corner[1] - 2 <= first_corner[1]:
                    if corner[1] < closest_quad_1[1]:
                        closest_quad_1 = corner
    if closest_quad_1[2] != 1:
        print("couldn't find rectangle: quad 1 missing")
        return rectangle
    rectangle.append(closest_quad_1)
    the_quad_2 = oriented_centroids[0]
    for corner in oriented_centroids:
        if corner[2] == 2:
            if corner[0] + 2 >= closest_quad_1[0] and corner[0] - 2 <= closest_quad_1[0]:
                if corner[1] + 2 >= closest_quad_3[1] and corner[1] - 2 <= closest_quad_3[1]:
                    the_quad_2 = corner
    if the_quad_2[2] != 2:
        print("couldn't find rectangle: quad 2 missing")
        return rectangle
    rectangle.append(the_quad_2)
    reordered_rectangle = [rectangle[0], rectangle[1], rectangle[3], rectangle[2]]
    return reordered_rectangle


def find_all_rectangles(oriented_centroids):
    rectangles = []
    # remaining_centroids = oriented_centroids
    for corner in oriented_centroids:
        if corner[2] == 4:
            new_rectangle = find_one_rectangle(oriented_centroids, corner)
            if not new_rectangle:
                print("couldn't find more rectangles")
                print(rectangles)
                return rectangles
            rectangles.append(new_rectangle)
            # remaining_centroids = [point for point in remaining_centroids if point not in new_rectangle]
    return rectangles


def draw_rectangles(img, oriented_centroids):
    rectangles_img = np.copy(img)
    rectangles = find_all_rectangles(oriented_centroids)
    for rect in rectangles:
        if len(rect) == 4:
            tuple_top_left = rc_to_xy(rect[0])
            tuple_bottom_right = rc_to_xy(rect[2])
            rectangles_img = cv2.rectangle(rectangles_img, tuple_top_left, tuple_bottom_right, (255, 0, 0), 3)
        if len(rect) == 3:
            all_the_points = np.array([list(rc_to_xy(point)) for point in rect]).reshape((-1, 1, 2))
            # print(all_the_points)
            rectangles_img = cv2.polylines(rectangles_img, all_the_points, True, (255, 0, 0), 3)  # type: ignore
    return rectangles_img


def run_a_test(img, blocksize, ksize, k):
    corners = get_corners_and_edges(img, blocksize, ksize, k)[0]
    centroids = find_centroids(corners)
    oriented_corners = centroids_and_orientations(img, centroids)
    circled_corners = make_circles(img, centroids)
    oriented_circled_corners = make_orientation_marks(circled_corners, oriented_corners)
    display(oriented_circled_corners)
    rectangles = draw_rectangles(img, oriented_corners)
    display(rectangles)


def run_all_the_tests(img):
    for blocksize in range(5, 8):
        for ksize in [3, 5, 7]:
            k = 0.06
            print([blocksize, ksize, k])
            run_a_test(img, blocksize, ksize, k)


# things that work:
# 4, 7. 0.06
# 5, 3, 0.06
# 5, 5, 0.06
# 5, 7, 0.06
# 6, 3, 0.06
# 6, 7, 0.06
# 7, 3, 0.06
# 7, 5, 0.06

if __name__ == "__main__":
    image_path = "./my_project/real_shapes.png"
    image = load(image_path)
    image_features = colorize_corners_and_edges(image)
    display(image)
    display(image_features)
    display(image + image_features)
