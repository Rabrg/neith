import numpy as np
import operator

from skimage import measure
from skimage import transform

from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

CONTOUR_LEVEL = 0.8


def equation_char_list(r):
    contours = measure.find_contours(r, CONTOUR_LEVEL)

    resized_char_dict = dict()
    for n, contour in enumerate(contours):
        min = np.min(contour, axis=0)
        max = np.max(contour, axis=0)
        resized_contour = transform.resize(r[int(min[0]):int(max[0]), int(min[1]):int(max[1])], (32, 32))
        resized_char_dict[min[1]] = resized_contour

    sorted1 = sorted(resized_char_dict.items(), key=operator.itemgetter(0))
    char_imgs = np.asarray([i[1] for i in sorted1])
    np.subtract(char_imgs, 0.5, out=char_imgs)
    return char_imgs


def get_iou(rec1, rec2):
    # determine the coordinates of the intersection rectangle
    x_left = max(rec1[0], rec2[0])
    y_top = max(rec1[1], rec2[1])
    x_right = min(rec1[2], rec2[2])
    y_bottom = min(rec1[3], rec2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    rec1_area = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    rec2_area = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(rec1_area + rec2_area - intersection_area)
    return iou
