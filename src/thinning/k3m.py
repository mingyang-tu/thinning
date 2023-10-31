import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate2d


WINDOW = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]], dtype=np.uint8)

LOOKUP = [
    [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60,
     62, 63, 96, 112, 120, 124, 126, 127, 129, 131, 135,
     143, 159, 191, 192, 193, 195, 199, 207, 223, 224,
     225, 227, 231, 239, 240, 241, 243, 247, 248, 249,
     251, 252, 253, 254],
    [7, 14, 28, 56, 112, 131, 193, 224],
    [7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135,
     193, 195, 224, 225, 240],
    [7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120,
     124, 131, 135, 143, 193, 195, 199, 224, 225, 227,
     240, 241, 248],
    [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120,
     124, 126, 131, 135, 143, 159, 193, 195, 199, 207,
     224, 225, 227, 231, 240, 241, 243, 248, 249, 252],
    [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120,
     124, 126, 131, 135, 143, 159, 191, 193, 195, 199,
     207, 224, 225, 227, 231, 239, 240, 241, 243, 248,
     249, 251, 252, 254]
]

ONEPIXEL = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56,
            60, 62, 63, 96, 112, 120, 124, 126, 127, 129, 131,
            135, 143, 159, 191, 192, 193, 195, 199, 207, 223,
            224, 225, 227, 231, 239, 240, 241, 243, 247, 248,
            249, 251, 252, 253, 254]


def k3m(binary: NDArray[np.uint8]):
    output = np.pad(binary, 1, mode="constant", constant_values=0)
    ROW, COL = output.shape

    while True:
        change = False

        borders = np.isin(correlate2d(output, WINDOW, mode="same"), LOOKUP[0]) & output.astype(np.bool_)

        for phase in range(1, 6):
            for i in range(1, ROW - 1):
                for j in range(1, COL - 1):
                    if borders[i, j]:
                        weight = np.sum(WINDOW * output[i - 1 : i + 2, j - 1 : j + 2])
                        if weight in LOOKUP[phase]:
                            output[i, j] = 0
                            change = True
        if not change:
            break

    for i in range(1, ROW - 1):
        for j in range(1, COL - 1):
            if output[i, j] > 0:
                weight = np.sum(WINDOW * output[i - 1 : i + 2, j - 1 : j + 2])
                if weight in ONEPIXEL:
                    output[i, j] = 0

    return output[1:-1, 1:-1]
