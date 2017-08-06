import numpy as np
import operator

from skimage import measure
from skimage import transform

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
