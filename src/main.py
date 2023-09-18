import cv2
import numpy as np
import matplotlib.pyplot as plt

from thinning import ArcelliBaja, k3m


def bgr2gray(image):
    cvt_mat = np.array([19/256, 183/256, 54/256], dtype=np.float64)
    return image.dot(cvt_mat)


def thresholding(image, threshold=220):
    return (image <= threshold).astype(np.uint8)


if __name__ == "__main__":
    image = cv2.imread(f"./data/1/database/base_1_1_1.bmp").astype(np.float64)
    binary = thresholding(bgr2gray(image))

    ab = ArcelliBaja(binary)
    out_ab = ab.run()
    out_k3m = k3m(binary)

    plt.figure()
    plt.imshow(binary, cmap="gray")
    plt.title("Original")

    plt.figure()
    plt.imshow(out_ab, cmap="gray")
    plt.title("Arcelli Baja")

    plt.figure()
    plt.imshow(out_k3m, cmap="gray")
    plt.title("K3M")
    plt.show()