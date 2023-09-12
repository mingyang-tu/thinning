import numpy as np
from numpy.typing import NDArray


def four_distance_transform(binary: NDArray[np.uint8]):
    ROW, COL = binary.shape
    output = binary.copy()

    for i in range(1, ROW):
        for j in range(1, COL):
            if output[i, j] != 0:
                output[i, j] = min(output[i, j-1], output[i-1, j]) + 1

    for i in range(ROW-2, -1, -1):
        for j in range(COL-2, -1, -1):
            if output[i, j] != 0:
                output[i, j] = min(
                    min(output[i, j+1], output[i+1, j]) + 1,
                    output[i, j]
                )

    return output
