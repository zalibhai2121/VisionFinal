from skimage import io, color
import os
import imghdr

source = r'C:\Users\Images\RGB'
destination = r'C:\Users\Images\Greyscale'

image_files = [os.path.join(root, filename)
                   for root, dirs, files in os.walk(source)
                   for filename in files
                   if imghdr.what(os.path.join(root, filename))]

for fn in image_files:
    rgb = io.imread(fn)
    grey = color.rgb2gray(rgb)
    head, tail = os.path.split(fn)
    io.imsave(os.path.join(destination, tail), grey)


""" BORROW FROM BELOW TO MAKE THIS ACTUALLY WORK"""

from numpy.core._multiarray_umath import ndarray

import hough_grid as hagrid
import cv2
import math
import random as rnd
import numpy as np
from collections import Counter
from display_img import display_img

# Constants
# Canny thresholds
THRESH_ONE = 10
THRESH_TWO = 40

# HoughLinesP parameters
RHO = 1
THETA = math.pi / 180
LINE_THRESH = 40
MIN_LENGTH = 50.0
MAX_GAP = 5.0

# Line drawing thickness
THICKNESS = 3

# Line drawing colors
BLACK = (255, 0, 0)

# Digit image dimensions
DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28
DIGIT_CHANNELS = 1

# Read image from disk. White = 255, Black = 0
img_orig = 255 - cv2.imread("sudoku_square/sudoku5.png", cv2.IMREAD_GRAYSCALE)
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1))


# Accepts a grayscale image and returns list of edges from Canny edge detector
def canny_edges(gray_img):
    """
    Takes a grayscale image, uses Canny edge detector to return list of edges
    :param gray_img: an image, grayscaled
    :return: list of edges from Canny edge detector
    """

    return cv2.Canny(gray_img, THRESH_ONE, THRESH_TWO)


def randcol():
    """
    Gives random colors
    :return: three random ints, in a tuple?
    """
    return (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255))


def length2(l):
    """
    Given a list of 4 elements, outputs the square of the length of a segment
    :param l: list of 4 elements
    :return: the square of the length of a segment (a, b) - (c, d)
    """

    a, b, c, d = l

    return ((b - a) * (b - a) + (d - c) * (d - c))


def line_intersection(lop):
    """
    Finds the interesction of the line through (a1, b1) and (c1, d1),
    and the line through (a2, b2) and (c2, d2)
    :param lop: list of 8 points
    :return: the point of intersection, (x,y) as a tuple
    TODO: WHY ARE WE USING TUPLES???
    """

    [a1, b1, c1, d1, a2, b2, c2, d2] = lop

    det = (d1 - b1) * (a2 - c2) - (a1 - c1) * (d2 - b2)
    det1 = (a1 * d1 - b1 * c1)
    det2 = (a2 * d2 - b2 * c2)
    x = (det1 * (a2 - c2) - det2 * (a1 - c1)) / det
    y = (det2 * (d1 - b1) - det1 * (d2 - b2)) / det

    return (x, y)


def add_lines(img, segs, color):
    """
    Draws segments (a, b), (c, d) in a list like [[a, b, c, d], [a, b, c, d], ...]
    atop a copy of the image img, using color. If color is -1, use random colors
    :param img: image
    :param segs: segments (a, b), (c, d)
    :param color:
    :return: the resulting image
    """

    result = img.copy()
    if len(img.shape) < 3:
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # color
    # hough = np.ones(edges.shape, dtype=np.uint8) # b/w

    # Loop through
    for i in range(0, len(segs)):
        l = segs[i][0]
        c = randcol() if color == -1 else color
        cv2.line(result, (l[0], l[1]), (l[2], l[3]), c, 3, cv2.LINE_AA)

    return result


def principal_angles(lines):
    """
    Takes a list of segments, returns a list in the same order as lines that clusters the angles
    Also returns list of labels sorted by frequency
    :param lines: list of line segments given by a hough transmform
    :return: a list of in the same order as lines that clusters the angles
    :return: also returns a list of labels sorted by frequency
    """

    # Build a list of angles for all the Hough segments
    angles = [-np.arctan((l[0][3] - l[0][1]) / (l[0][2] - l[0][0])) for l in lines]

    # Cluster them to find the principle angles
    num_centers = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, a_labels, a_centers = cv2.kmeans(np.asarray(angles, dtype='float32'), num_centers, \
                                          None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Build a dictionary mapping each label (0, 1, 2, ...) to its frequency in the list
    a_labels_map = Counter(a_labels.ravel().tolist())
    print("angle map", a_labels_map)
    kv = [(a_labels_map[i], i) for i in a_labels_map]
    kv.sort()
    print("KV", kv)

    return a_labels, [x[1] for x in kv]  # removed redundant parentheses around outside of stuff to be returned


def segment_bounds(segs, K):
    """
    Given a list of segments, fit them to an equally-spaced grid of K lines
    :param segs: list of segments all roughly at the same angle
    :param K: grid of K-lines, equally spaced?? Whatever that means
    :return: indices of two of the segments that bound the grid
    """

    # Make a list of radii for the segments having the most frequent label/angle
    # We use the fact that the equation of the line at angle theta and distance R from
    #  the origin has equation R = x cos(theta) + y sin(theta)
    # Build a list of angles for all the Hough segments
    angles = [-np.arctan((l[0][3] - l[0][1]) / (l[0][2] - l[0][0])) for l in segs]
    radii_idx = [(segs[i][0][0] * np.sin(angles[i]) + segs[i][0][1] * np.cos(angles[i]), i) \
                 for i in range(len(segs))]
    radii_idx.sort()
    print(radii_idx)
    radii = [r[0] for r in radii_idx]

    # Now try to fit the segments to the best ten equally-spaced radii
    minbadness = float("inf")
    bestpair = (0, 0)

    for i in range(len(radii)):
        for j in range(i + K - 1, len(radii)):
            step = (radii[j] - radii[i]) / (K - 1)
            badness = 0
            for k in range(len(radii)):
                seg = radii[k]
                thisbad = 0
                if seg <= radii[i]:
                    thisbad = radii[i] - seg
                elif seg >= radii[j]:
                    thisbad = seg - radii[j]
                else:
                    thisbad = (seg - radii[i]) % step
                    thisbad = min(thisbad, step - thisbad)
                    thisbad = thisbad * thisbad  # square penalty for inside the grid
                # print(step, thisbad)
                badness += thisbad * length2(segs[radii_idx[k][1]][0])
            if badness < minbadness:
                minbadness = badness
                bestpair = (i, j)

    return radii_idx[bestpair[0]][1], radii_idx[bestpair[1]][1]  # removed redundant parentheses


def img_prep(img):
    """
    For smoothing and adaptive thresholding
    :param img: the image
    :return: an image, smoothed then gaussian blurred
    """
    res = img.copy()  # res = result aka the resulting image

    # for smoothing
    # we smooth first because adaptive thresholding without smoothing means broken digits
    # kernel sizes must be positive and odd and the kernel must be square- somewhere between (5,5) and (9,9)
    # res = cv2.GaussianBlur(res, (3,3), 0) #smoothes sharp parts, takes care of jagged edge clutter #else 7,7

    # adaptive thresholding using nearest 30 pixel neighbors
    # should be somewhere between 11-30
    # last value no more than 3
    # not 9, 3
    res = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)  # or 2

    # Reverse the image to so the white text is found when looking for the contours
    res = inverseColors(res)
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1))
    res = cv2.dilate(res, elem)
    return res


def inverseColors(img):
    """
    Inverses the colors of an image
    :param img: an image
    :return: a color inversed image
    """
    img = (255 - img)
    return img


def getContours(img):
    """
    find and sort contours
    :param img: an image
    :return:
    """

    res = img.copy()
    inverseColors(res)
    img_prep(res)

    external_contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(external_contours, key=cv2.contourArea)
    square = contours[-1]  # returning last value bc sorted in ascending order, we want largest square

    return square


def findCorners(img):
    """
    Find the corners of the largest square
    :param img: an image
    :return: an array of the corner coordinates
    """
    im = img.copy()
    board = getContours(im)  # find largest square

    # using arrays of 1 coordinate
    right = [cornerPoint[0][0] + cornerPoint[0][1] for cornerPoint in board] #00 for x, 01 for y
    left = [cornerPoint[0][0] - cornerPoint[0][1] for cornerPoint in board]

    el = lambda s: (s[1])  # fetching first item out of iterable object #WHY DOESN'T IT WORK AT INDEX 0?!?!?!

    # Bottom-right point has the largest (x + y) value
    bottomRight, im = max(enumerate(right), key=el)  # compare each item in right list by the value at index 1

    # Top-left has point smallest (x + y) value
    topLeft, im = min(enumerate(right), key=el)

    # Bottom-left point has smallest (x - y) value
    bottomLeft, im = min(enumerate(left), key=el)

    # Top-right point has largest (x - y) value
    topRight, im = max(enumerate(left), key=el)

    # Want to return an array of all 4 coordinates, with every point in its own array
    return [board[topLeft][0], board[topRight][0], board[bottomRight][0], board[bottomLeft][0]]


def placeCircles(img, cornerPoints):
    '''
    Draws circles on the image
    :param img: an image
    :param cornerPoints: the corner points as an array of arrays, obtained from findCorners
    :return: the image with circles on it
    '''
    im = img.copy()
    color = (0, 0, 255)
    radius = 4
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    for pt in cornerPoints:
        # image = cv2.circle(image, center_coordinates, radius, color, thickness)
        im = cv2.circle(im, tuple(int(x) for x in pt), radius, color, -1)  # 1 for hollow

    display_img(im)
    return im


def boundsFunc(img):
    """
    A function that exists purely to present the tuples
    in the right format so the grader can read
    worked fine before, but the grader wasn't happy with formatting
    :param img: an image
    :return: the correctly-formatted tuple Corners
    """

    display_img(img)
    checkptimg = img_prep(img)
    display_img(checkptimg)
    image = img
    prep = img_prep(image)
    corners = findCorners(prep)
    tupleCorners = [(corners[0][0], corners[0][1]), (corners[1][0], corners[1][1]), (corners[2][0], corners[2][1]),
                    (corners[3][0], corners[3][1])]
    return (tupleCorners)


"""
def sudoku_bounds(img):

    #Finds the four segments bounding a sudoku board img
    #:param img: a sudoku board
    #:return:


    cv2.imshow("Orig", img) #show the image

    # Canny edge detection
    edges = canny_edges(img)
    cv2.imshow("Canny", edges)

    # Perform the line detection
    lines = cv2.HoughLinesP(edges, RHO, THETA, LINE_THRESH,\
        minLineLength=MIN_LENGTH, maxLineGap=MAX_GAP)
    cv2.imshow("Hough", add_lines(img, lines, -1))

    # Find the principle angles
    a_labels, freqs = principal_angles(lines)
    lines0 = [lines[i] for i in range(len(lines)) if a_labels[i] == freqs[2]]
    lines1 = [lines[i] for i in range(len(lines)) if a_labels[i] == freqs[1]]
    imga = add_lines(img, lines0, (255, 0, 0))
    imga = add_lines(imga, lines1, (0, 0, 255))
    cv2.imshow("Two Max Angles", imga)

    # Find bounding lines for angle0
    l0, l1 = [lines0[i] for i in segment_bounds(lines0, 10)]
    m0, m1 = [lines1[i] for i in segment_bounds(lines1, 10)]
    bounds_img = add_lines(img, [l0, l1, m0, m1], (255, 255, 0))

    # Find the four corners
    c0 = line_intersection(l0[0].ravel().tolist() + m0[0].ravel().tolist())
    c1 = line_intersection(l0[0].ravel().tolist() + m1[0].ravel().tolist())
    c2 = line_intersection(l1[0].ravel().tolist() + m1[0].ravel().tolist())
    c3 = line_intersection(l1[0].ravel().tolist() + m0[0].ravel().tolist())

    cv2.circle(bounds_img, tuple(map(int, c0)), 5, (255, 0, 255), 5)
    cv2.circle(bounds_img, tuple(map(int, c1)), 5, (255, 0, 255), 5)
    cv2.circle(bounds_img, tuple(map(int, c2)), 5, (255, 0, 255), 5)
    cv2.circle(bounds_img, tuple(map(int, c3)), 5, (255, 0, 255), 5)
    cv2.imshow("Bounds", bounds_img)
"""

"""
#Video capture from webcam
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sudoku_bounds(gray)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
cap.release()

cv2.waitKey(0) # Wait for any key to be pressed (with the image window active)
cv2.destroyAllWindows() #Close all windows
"""
