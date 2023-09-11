import cv2
import numpy as np
import matplotlib.pyplot as plt

from thinning.distance_transform import inner_contour
from thinning.utils import bgr2gray, thresholding


if __name__ == "__main__":
    image = cv2.imread("./data/word_1_1_1.bmp").astype(np.float64)
    binary = thresholding(bgr2gray(image))
    ic = inner_contour(binary)
    plt.imshow(ic, cmap="gray")
    plt.show()
