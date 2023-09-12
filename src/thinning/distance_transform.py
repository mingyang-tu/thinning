import numpy as np
from numpy.typing import NDArray


def four_distance_transform(binary: NDArray[np.uint8]):
    ROW, COL = binary.shape
    binary_copy = binary.copy()

    for i in range(1, ROW):
        for j in range(1, COL):
            if binary_copy[i, j] != 0:
                binary_copy[i, j] = min(binary_copy[i, j-1], binary_copy[i-1, j]) + 1

    for i in range(ROW-2, -1, -1):
        for j in range(COL-2, -1, -1):
            if binary_copy[i, j] != 0:
                binary_copy[i, j] = min(
                    min(binary_copy[i, j+1], binary_copy[i+1, j]) + 1,
                    binary_copy[i, j]
                )

    return binary_copy
