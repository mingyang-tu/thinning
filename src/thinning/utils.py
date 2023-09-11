import numpy as np


def bgr2gray(image):
    cvt_mat = np.array([19/256, 183/256, 54/256], dtype=np.float64)
    return image.dot(cvt_mat)


def thresholding(image, threshold=220):
    return (image <= threshold).astype(np.uint8)
