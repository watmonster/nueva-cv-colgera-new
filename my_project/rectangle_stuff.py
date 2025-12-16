#!/usr/bin/env python

# the comments noqa and type: ignore are not relevant they just make flake8 and pylance shut up when they're wrong

from skimage import io  # type: ignore
from skimage import color  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import cv2


plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def load(image_path):  # load images in a way that makes things happy
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
    # I'm using list [row,col] as a convention but sometimes cv2 likes that and sometimes it likes tuple (x,y)
    # so this does [row,col] -> (x,y)
    return (int(round(coords_list[1])), int(round(coords_list[0])))


def corner_detection(img, blocksize, ksize, k):  # cornerHarris and helpers
    gray_img = make_grayscale(img)  # cornerHarris requires grayscale images
    corners = np.copy(gray_img)  # cv2 often changes the image it's given, so we copy so it doesn't overwrite
    cv2.cornerHarris(gray_img, blocksize, ksize, k, corners)  # noqa
    return corners


def clamp(img):  # makes images all either 0 or 1, no negatives, no nonsense
    clamped_img = np.copy(img)
    clamped_img[clamped_img < np.max(clamped_img) / 20] = 0  # 20 is kinda arbitrary it just seems to work nicely
    clamped_img[clamped_img >= np.max(clamped_img) / 20] = 1
    return clamped_img


def get_corners_and_edges(img, blocksize, ksize, k):
    corners = corner_detection(img, blocksize, ksize, k)  # so funny story
    # turns out harris corner detection also runs edge detection
    # but the pixels it identifies as corners are positive, and edges are negative
    # so we can use clamp to deal with that, yay
    clamped_corners = clamp(corners)
    clamped_edges = clamp(corners * (-1))
    return clamped_corners, clamped_edges


def colorize_corners_and_edges(img, blocksize, ksize, k):  # for testing/human-readability purposes
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
    total_brightness = 0
    pix_loc_row = int(round(pix_loc[0]))
    pix_loc_col = int(round(pix_loc[1]))
    row_mult = quads[quad_num][0]
    col_mult = quads[quad_num][1]  # setting these up for the loop we're about to run
    img_shape = img.shape
    img_num_rows = img_shape[0]
    img_num_cols = img_shape[1]  # these are to deal with index errors in case we try to iterate outside the image
    for row in range(3, radius + 1):
        for col in range(3, radius + 1):
            check_row = int(pix_loc_row + row * row_mult)
            check_col = int(pix_loc_col + col * col_mult)  # what [row, col] are we getting the brightness of?
            if check_row >= 0 and check_col >= 0 and check_row < img_num_rows - 1 and check_col < img_num_cols - 1:
                # if we're inside the image
                current_brightness = gray_clamped_img[check_row, check_col]
                total_brightness += current_brightness  # increment based on the current pixel
    return total_brightness / (radius**2)  # average


def find_corner_orientation(img, corner_loc, radius=5):  # finding which way a corner is pointing
    quad_1_brightness = avg_pix_brightness(img, corner_loc, 1, radius)
    quad_2_brightness = avg_pix_brightness(img, corner_loc, 2, radius)
    quad_3_brightness = avg_pix_brightness(img, corner_loc, 3, radius)
    quad_4_brightness = avg_pix_brightness(img, corner_loc, 4, radius)  # find the brightness of each quadrant
    brightness_list = [quad_1_brightness, quad_2_brightness, quad_3_brightness, quad_4_brightness]
    max_brightness = max(brightness_list)  # whichever one is the most is probably the one that corner is pointing at
    max_quad = brightness_list.index(max_brightness) + 1  # quadrants are 1-indexed
    return max_quad


def find_centroids(harris_result):  # harris gives blobs; I want points
    # step1 = cv2.dilate(harris_result, None, iterations=5)  # type: ignore # makes the blobs bigger/more uniform
    # _, step2 = cv2.threshold(step1, 0.01 * np.max(step1), 255, 0)  # makes it all 0 or 1 because cv2 likes that here
    _, step2 = cv2.threshold(harris_result, 0.01 * np.max(harris_result), 255, 0)  # makes it all 0 or 1 for cv2 reasons
    step3 = np.uint8(step2)  # more fun with image types... ugh
    centroids = cv2.connectedComponentsWithStats(step3)[3]  # type: ignore # finds the centroids of blobs
    # connectedComponentsWithStats actually gives a bunch of information but only [3] is relevant here
    revised_centroids = centroids[1:]  # assume the centroid of the background is first?? (we think)
    # cv2 is inconsistent yet again (output is in (x,y) form) so we gotta flip it and make it a list instead of a tuple
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
    rectangle = []  # it's just a list of points
    rectangle.append(first_corner)
    closest_quad_3 = [0, 0, 0]  # just give it a value of the correct type
    for corner in oriented_centroids:
        if corner[2] == 3:  # if we do have a functional quad 3 corner
            if corner[1] > first_corner[1]:
                if corner[0] + 4 >= first_corner[0] and corner[0] - 4 <= first_corner[0]:
                    if closest_quad_3[0] != 0:
                        if corner[0] < closest_quad_3[0]:  # but the current corner is better
                            closest_quad_3 = corner  # then replace it
                    else:
                        closest_quad_3 = corner
    if closest_quad_3[0] != 0:
        rectangle.append(closest_quad_3)  # we're fine; add it to the rectangle
    closest_quad_1 = [0, 0, 0]  # same process for quad 1 except we're moving down instead of right
    for corner in oriented_centroids:
        if corner[2] == 1:
            if corner[0] > first_corner[0]:
                if corner[1] + 4 >= first_corner[1] and corner[1] - 4 <= first_corner[1]:
                    if closest_quad_1[0] != 0:
                        if corner[1] < closest_quad_1[1]:
                            closest_quad_1 = corner
                    else:
                        closest_quad_1 = corner
    if closest_quad_1[0] != 0:
        rectangle.append(closest_quad_1)  # etc etc
    if len(rectangle) < 2:  # if we only have a quad 4 point, or somehow an empty list
        return []
    if len(rectangle) == 3:  # rectangles only require 3 points but 4 is nice
        the_quad_2 = [0, 0, 0]  # there should only be one that works at this point
        for corner in oriented_centroids:
            if corner[2] == 2:
                if corner[0] + 4 >= closest_quad_1[0] and corner[0] - 4 <= closest_quad_1[0]:
                    if corner[1] + 4 >= closest_quad_3[1] and corner[1] - 4 <= closest_quad_3[1]:
                        the_quad_2 = corner  # if it's the one, then it's the one
                        rectangle.append(the_quad_2)
                        reordered_rectangle = [rectangle[0], rectangle[1], rectangle[3], rectangle[2]]
                        return reordered_rectangle  # so we can draw it easily later
        return rectangle  # if we end up with quads 4, 3, 1
    if len(rectangle) == 2:  # possibly at this point we can still find a quad 2 and get 3 points for a rectangle
        closest_quad_2 = [0, 0, 0]
        for corner in oriented_centroids:
            if corner[2] == 2:
                if corner[0] > first_corner[0] and corner[1] > first_corner[1]:  # correct positioning
                    if rectangle[1][2] == 3:  # further steps depend on what info we currently have
                        if corner[1] + 4 >= rectangle[1][1] and corner[1] - 4 <= rectangle[1][1]:
                            if closest_quad_2[0] != 0:  # if we already have a quad 2 corner
                                if corner[0] < closest_quad_2[0]:
                                    closest_quad_2 = corner
                            else:  # if we don't already have a quad 2 corner then assign it
                                closest_quad_2 = corner
                    elif rectangle[1][2] == 1:
                        if corner[0] + 4 >= rectangle[1][0] and corner[0] - 4 <= rectangle[1][0]:
                            if closest_quad_2[0] != 0:
                                if corner[1] < closest_quad_2[1]:
                                    closest_quad_2 = corner
                            else:
                                closest_quad_2 = corner
        if closest_quad_2[0] != 0:
            rectangle.append(closest_quad_2)
            return rectangle
    # sad :( (at this point we have 2 points, rectangles require 3)
    return []


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


def draw_rectangles(img, rectangles):  # let's see where it thinks the rectangles are
    rectangles_img = np.copy(img)  # ARGHHH NOT AGAIN
    green_rectangles = []
    # rectangles = find_all_rectangles(oriented_centroids)
    for rect in rectangles:
        if len(rect) == 4:  # if it's a rectangle
            tuple_top_left = rc_to_xy(rect[0])
            tuple_bottom_right = rc_to_xy(rect[2])  # then we can just make a rectangle
            if tuple_bottom_right[0] - tuple_top_left[0] <= 100 and tuple_bottom_right[1] - tuple_top_left[1] <= 200:
                rectangles_img = cv2.rectangle(rectangles_img, tuple_top_left, tuple_bottom_right, (0, 255, 0), 3)
                green_rectangles.append(rect)
            else:
                rectangles_img = cv2.rectangle(rectangles_img, tuple_top_left, tuple_bottom_right, (255, 0, 0), 3)
        if len(rect) == 3:  # if it's a triangle
            rect_quads = [rect[0][2], rect[1][2], rect[2][2]]
            if rect_quads == [4, 3, 1]:
                tuple_top_left = rc_to_xy(rect[0])
                tuple_bottom_right = rc_to_xy([rect[2][0], rect[1][1]])
                if tuple_bottom_right[0] - tuple_top_left[0] <= 100 and tuple_bottom_right[1] - tuple_top_left[1] <= 200:  # noqa
                    rectangles_img = cv2.rectangle(rectangles_img, tuple_top_left, tuple_bottom_right, (0, 255, 0), 3)
                    green_rectangles.append(rect)
                else:
                    rectangles_img = cv2.rectangle(rectangles_img, tuple_top_left, tuple_bottom_right, (255, 0, 0), 3)
                # rectangles_img = cv2.rectangle(rectangles_img, tuple_top_left, tuple_bottom_right, (0, 255, 0), 3)
            if 4 in rect_quads and 2 in rect_quads:
                tuple_top_left = rc_to_xy(rect[0])
                tuple_bottom_right = rc_to_xy(rect[2])
                if tuple_bottom_right[0] - tuple_top_left[0] <= 100 and tuple_bottom_right[1] - tuple_top_left[1] <= 200:  # noqa
                    rectangles_img = cv2.rectangle(rectangles_img, tuple_top_left, tuple_bottom_right, (0, 255, 0), 3)
                    green_rectangles.append(rect)
                else:
                    rectangles_img = cv2.rectangle(rectangles_img, tuple_top_left, tuple_bottom_right, (255, 0, 0), 3)
                # rectangles_img = cv2.rectangle(rectangles_img, tuple_top_left, tuple_bottom_right, (0, 255, 0), 3)
    return rectangles_img, green_rectangles


def run_a_test(img, blocksize, ksize, k):
    # this is for testing different param sets for corner detection
    # it draws the things and then shows them to me fast
    corners = get_corners_and_edges(img, blocksize, ksize, k)[0]
    centroids = find_centroids(corners)
    oriented_corners = centroids_and_orientations(img, centroids)
    # circled_corners = make_circles(img, centroids)
    # oriented_circled_corners = make_orientation_marks(circled_corners, oriented_corners)
    # display(oriented_circled_corners)
    rectangles = find_all_rectangles(oriented_corners)
    rectangles_img, green_rectangles = draw_rectangles(img, rectangles)
    # display(rectangles_img)
    return green_rectangles


def combine_params(img):
    all_green_rectangles = []
    for blocksize in range(6, 8):  # this range seems to be good
        for ksize in [5, 7]:  # same here
            k = 0.06   # turns out this works well! why? nobody knows...
            green_rectangles = run_a_test(img, blocksize, ksize, k)  # do the thing
            all_green_rectangles += green_rectangles
    green_rectangles_img = draw_rectangles(img, all_green_rectangles)[0]
    display(green_rectangles_img)
    return all_green_rectangles


def rects_and_corners(img, blocksize=6, ksize=5, k=0.06):
    corners = get_corners_and_edges(img, blocksize, ksize, k)[0]
    centroids = find_centroids(corners)
    oriented_corners = centroids_and_orientations(img, centroids)
    all_green_rectangles = run_a_test(img, blocksize, ksize, k)
    print(all_green_rectangles)
    green_rectangles_list = [all_green_rectangles[i][j] for i in range(len(all_green_rectangles)) for j in range(len(all_green_rectangles[i]))]  # noqa  # type: ignore
    print(green_rectangles_list)
    not_rectangle_corners = [oriented_corners[i] for i in range(len(oriented_corners)) if oriented_corners[i] not in green_rectangles_list]  # noqa
    circled_corners = make_circles(img, not_rectangle_corners)
    oriented_circled_corners = make_orientation_marks(circled_corners, not_rectangle_corners)
    # display(oriented_circled_corners)
    rectangles_oriented_corners = draw_rectangles(oriented_circled_corners, all_green_rectangles)[0]
    display(rectangles_oriented_corners)


if __name__ == "__main__":
    img_path = "./my_project/column_0004.png"
    img = load(img_path)[:, :, 0:3]
    rects_and_corners(img)
