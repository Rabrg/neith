import numpy as np
import operator

from skimage import measure
from skimage import transform

from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

IOU_THRESHOLD = 0
CONTOUR_LEVEL = 0.8
MIN_CONTOUR_SIZE = 8


def remove_overlap_contours(contours):
    to_remove = []
    for i1, contour1 in enumerate(contours):
        min = np.min(contour1, axis=0)
        max = np.max(contour1, axis=0)
        rec1 = Rectangle(min[1], min[0], max[1], max[0])
        area1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        for i2, contour2 in enumerate(contours):
            min = np.min(contour2, axis=0)
            max = np.max(contour2, axis=0)
            rec2 = Rectangle(min[1], min[0], max[1], max[0])
            area2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            if contour1 is not contour2:
                iou = get_iou(rec1, rec2)
                if iou > IOU_THRESHOLD:
                    if area1 > area2 and i2 not in to_remove:
                        to_remove.append(i2)
                    elif i1 not in to_remove:
                        to_remove.append(i1)
    contours_new = []
    for i, c in enumerate(contours):
        if i not in to_remove:
            contours_new.append(c)
    return contours_new


def equation_char_list(r):
    contours = measure.find_contours(r, CONTOUR_LEVEL)
    contours = remove_overlap_contours(contours)

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
