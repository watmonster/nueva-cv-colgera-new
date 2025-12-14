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


def load(image_path):  # load images in a way that makes most things happy
    out = io.imread(image_path)
    out = out.astype(np.float64) / 255
    return out


def display(img):
    # Show image
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def make_grayscale(img):  # make the image grayscale, deal with types
    return color.rgb2gray(img).astype("float32")


def rc_to_xy(coords_list):
    # deal with cv2's inconsistencies
    # I'm using [row,col] as a convention but sometimes cv2 likes that and sometimes it likes (x,y))
    # so this  makes [row,col] -> (x,y)
    return (int(round(coords_list[1])), int(round(coords_list[0])))


def corner_detection(img, blocksize=3, ksize=5, k=0.04):  # cornerHarris and helpers
    gray_img = make_grayscale(img)  # cornerHarris requires grayscale images
    corners = np.copy(gray_img)  # cv2 often changes the image it's given, so we copy so it doesn't overwrite
    cv2.cornerHarris(gray_img, blocksize, ksize, k, corners)  # noqa
    return corners


def clamp(img):  # makes images all either 0 or 1, no negatives, no nonsense
    clamped_img = np.copy(img)
    clamped_img[clamped_img < np.max(clamped_img) / 20] = 0  # 20 is kinda arbitrary it just seems to work nicely
    clamped_img[clamped_img >= np.max(clamped_img) / 20] = 1
    return clamped_img


def get_corners_and_edges(img, blocksize=3, ksize=5, k=0.04):
    corners = corner_detection(img, blocksize, ksize, k)  # so funny story
    # turns out harris corner detection also runs edge detection
    # but the pixels it identifies as corners are positive, and edges are negative
    # so we can use clamp to deal with that, yay
    clamped_corners = clamp(corners)
    clamped_edges = clamp(corners * (-1))
    return clamped_corners, clamped_edges


def colorize_corners_and_edges(img, blocksize=3, ksize=5, k=0.04):  # for testing/human-readability purposes
    corners, edges = get_corners_and_edges(img, blocksize, ksize, k)
    green_blue_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)  # make another image
    # dilate makes the corners big so we can see them
    # corners are green because I said so
    green_blue_image[:, :, 1] = cv2.dilate(corners, None, iterations=5)  # type: ignore
    # edges are blue
    green_blue_image[:, :, 2] = edges
    return green_blue_image


def load_real_image():  # this and load_test_image are for my convenience
    return load("./my_project/real_shapes.png")


def load_test_image():
    return load("./my_project/test_shapes.png")[:, :, 0:3]  # slicing removes alpha channel because cv2 doesn't like it


quads = {1: [-1, 1], 2: [-1, -1], 3: [1, -1], 4: [1, 1]}  # [row, col] increments for iterating over each quadrant


def avg_pix_brightness(img, pix_loc, quad_num, radius):
    # this helps with finding the corner orientation
    # it finds the average pixel brightness in a certain radius (taxicab) in a certain quadrant related to a given pixel
    gray_img = make_grayscale(img)
    gray_clamped_img = clamp(gray_img)  # I just want 0s and 1s because the real images are a bit finicky
    # display(gray_clamped_img)
    total_brightness = 0
    pix_loc_row = int(round(pix_loc[0]))
    pix_loc_col = int(round(pix_loc[1]))
    row_mult = quads[quad_num][0]
    col_mult = quads[quad_num][1]  # setting these up for the loop we're about to run
    for row in range(radius):
        for col in range(radius):
            if row != 0 and col != 0:  # don't include the row and col of the pixel itself because that seems to help
                check_row = int(pix_loc_row + row * row_mult)
                check_col = int(pix_loc_col + col * col_mult)  # what [row, col] are we getting the brightness of?
                current_brightness = gray_clamped_img[check_row, check_col]
                total_brightness += current_brightness  # increment as appropriate
    return total_brightness / (radius**2)  # I asked for average; I don't know if radius^2 is correct but eh it works


def find_corner_orientation(img, corner_loc, radius=10):  # finding which way a corner is pointing
    quad_1_brightness = avg_pix_brightness(img, corner_loc, 1, radius)
    quad_2_brightness = avg_pix_brightness(img, corner_loc, 2, radius)
    quad_3_brightness = avg_pix_brightness(img, corner_loc, 3, radius)
    quad_4_brightness = avg_pix_brightness(img, corner_loc, 4, radius)  # find the brightness of each quadrant
    brightness_list = [quad_1_brightness, quad_2_brightness, quad_3_brightness, quad_4_brightness]
    max_brightness = max(brightness_list)  # whichever one is the most is probably the one that corner is pointing at
    max_quad = brightness_list.index(max_brightness) + 1  # quadrants are 1-indexed
    return max_quad


def find_centroids(harris_result):  # harris gives blobs; I want points
    step1 = cv2.dilate(harris_result, None, iterations=5)  # type: ignore # makes the blobs bigger/more uniform
    _, step2 = cv2.threshold(step1, 0.01 * np.max(step1), 255, 0)  # makes it all 0 or 1 because cv2 likes that here
    step3 = np.uint8(step2)  # more fun with image types... ugh
    centroids = cv2.connectedComponentsWithStats(step3)[3]  # type: ignore # finds the centroids of blobs
    # connectedComponentsWithStats actually gives a bunch of information but only [3] is relevant here
    revised_centroids = centroids[1:]  # assume the centroid of the background is first?? (we think)
    # cv2 is inconsistent yet again so we gotta flip it and make it a list instead of a tuple
    flipped_centroids = [[float(centroid[1]), float(centroid[0])] for centroid in revised_centroids]
    return flipped_centroids


def centroids_and_orientations(img, centroids, radius=5):  # find the orientations of all of the corners
    centroids_and_quads = []
    for centroid in centroids:
        # make them [row, col, quad] because that's how I decided we're doing this
        centroids_and_quads.append([centroid[0], centroid[1], find_corner_orientation(img, centroid, radius)])
    return centroids_and_quads


def make_circles(img, pixel_locs):  # draw some circles so that hooman can understand output
    circles_img = np.copy(img)  # more cv2 annoyance
    for pix in pixel_locs:
        tuple_pix = rc_to_xy(pix)  # and some more! yay! I love cv2!!!
        circles_img = cv2.circle(circles_img, tuple_pix, 20, (255, 0, 0), 3)
    return circles_img


def make_orientation_marks(img, pixel_locs_orientations):  # draw lines theoretically pointing where the corners face
    orientations_img = np.copy(img)  # YAY CV2 (can you tell this annoys me)
    for pix in pixel_locs_orientations:
        # if the corner is like _| the line should go \ if it's like |_ the line should go / etc
        tuple_pix = rc_to_xy(pix)
        line_end = (tuple_pix[0] + quads[pix[2]][1] * 30, tuple_pix[1] + quads[pix[2]][0] * 30)
        orientations_img = cv2.line(orientations_img, tuple_pix, line_end, (255, 0, 0), 5)
    return orientations_img


def find_one_rectangle(oriented_centroids, first_corner):  # noqa
    # probably the most incomprehensible function in this file
    # high-level: start with the first corner, quadrant 4
    # go to the right to find a quadrant 3 corner
    # go down from the first corner to find a quadrant 1 corner
    # reference the 2nd and 3rd corners to find the last quad 2 corner
    # print(oriented_centroids)
    rectangle = []  # it's just a list of points
    rectangle.append(first_corner)
    print(first_corner)
    closest_quad_3 = oriented_centroids[0]  # just give it a value of the correct type
    for corner in oriented_centroids:
        if corner[2] == 3 and closest_quad_3[2] != 3:  # if we don't have a functional quad 3 corner yet
            if corner[1] > first_corner[1]:
                if corner[0] + 2 >= first_corner[0] and corner[0] - 2 <= first_corner[0]:
                    closest_quad_3 = corner  # make there be a functional quad 3 corner
        elif corner[2] == 3:  # if we do have one
            if corner[1] > first_corner[1]:
                if corner[0] + 2 >= first_corner[0] and corner[0] - 2 <= first_corner[0]:
                    if corner[0] < closest_quad_3[0]:  # but the current corner is better
                        closest_quad_3 = corner  # then replace it
    if closest_quad_3[2] != 3:  # if we couldn't find a single one that works no matter how badly
        print("couldn't find rectangle: quad 3 missing")
        print(rectangle)
        return rectangle  # then stop already
    rectangle.append(closest_quad_3)  # but otherwise we're fine; add it to the rectangle
    closest_quad_1 = oriented_centroids[0]  # same process for quad 1 except we're moving down instead of right
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
        print(rectangle)
        return rectangle
    rectangle.append(closest_quad_1)  # etc etc
    the_quad_2 = oriented_centroids[0]  # there should only be one that works at this point
    for corner in oriented_centroids:
        if corner[2] == 2:
            if corner[0] + 2 >= closest_quad_1[0] and corner[0] - 2 <= closest_quad_1[0]:
                if corner[1] + 2 >= closest_quad_3[1] and corner[1] - 2 <= closest_quad_3[1]:
                    the_quad_2 = corner  # if it's the one, then it's the one
                    break
    if the_quad_2[2] != 2:  # same thing here
        print("couldn't find rectangle: quad 2 missing")
        print(rectangle)
        return rectangle
    rectangle.append(the_quad_2)
    reordered_rectangle = [rectangle[0], rectangle[1], rectangle[3], rectangle[2]]  # so we can draw it nicely later
    return reordered_rectangle


def find_all_rectangles(oriented_centroids):  # this is just math, no CV, so we don't need the image
    rectangles = []
    # remaining_centroids = oriented_centroids
    for corner in oriented_centroids:
        if corner[2] == 4:  # for every quad 4 corner see if it's got a rectangle
            new_rectangle = find_one_rectangle(oriented_centroids, corner)
            if new_rectangle:  # because empty lists are sad and annoying
                rectangles.append(new_rectangle)
            # remaining_centroids = [point for point in remaining_centroids if point not in new_rectangle]
    return rectangles


def draw_rectangles(img, oriented_centroids):  # let's see where it thinks the rectangles are
    rectangles_img = np.copy(img)  # ARGHHH NOT AGAIN
    rectangles = find_all_rectangles(oriented_centroids)
    for rect in rectangles:
        if len(rect) == 4:  # if it's a rectangle
            tuple_top_left = rc_to_xy(rect[0])
            tuple_bottom_right = rc_to_xy(rect[2])  # then we can just make a rectangle
            rectangles_img = cv2.rectangle(rectangles_img, tuple_top_left, tuple_bottom_right, (255, 0, 0), 3)
        if len(rect) == 3:  # if it's a triangle
            # this line is magic, I don't know why it works, but it's what the documentation did so I won't question it
            all_the_points = np.array([list(rc_to_xy(point)) for point in rect]).reshape((-1, 1, 2))
            # polylines just makes a polygon with some points, yay
            rectangles_img = cv2.polylines(rectangles_img, all_the_points, True, (255, 0, 0), 3)  # type: ignore
    return rectangles_img


def run_a_test(img, blocksize, ksize, k):
    # this is for testing different param sets for corner detection
    # it draws the things and then shows them to me fast
    corners = get_corners_and_edges(img, blocksize, ksize, k)[0]
    centroids = find_centroids(corners)
    oriented_corners = centroids_and_orientations(img, centroids)
    circled_corners = make_circles(img, centroids)
    oriented_circled_corners = make_orientation_marks(circled_corners, oriented_corners)
    display(oriented_circled_corners)
    rectangles = draw_rectangles(img, oriented_corners)
    display(rectangles)


def run_all_the_tests(img):
    for blocksize in range(5, 8):  # this range seems to be good but I haven't narrowed it down further
        for ksize in [3, 5, 7]:  # same here
            print("----------")
            k = 0.06   # turns out this works well! why? nobody knows...
            print([blocksize, ksize, k])  # so I know what I'm looking at
            run_a_test(img, blocksize, ksize, k)  # do the thing


if __name__ == "__main__":
    image_path = "./my_project/real_shapes.png"
    image = load(image_path)
    image_features = colorize_corners_and_edges(image)
    display(image)
    display(image_features)
    display(image + image_features)
