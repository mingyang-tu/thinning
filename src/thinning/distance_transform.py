import numpy as np
import cv2


def inner_contour(binary):      # input / output : np.uint8
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=1)
    # 0: background, 1: contour, 2: inner
    return binary + erosion
