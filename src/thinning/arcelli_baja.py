import numpy as np

from .distance_transform import inner_contour


# FIXME: isolated point
# TODO: change into class
def arcelli_baja(binary):       # input : np.uint8
    ROW, COL = binary.shape
    inner = binary.copy()
    output = np.zeros((ROW, COL), dtype=np.uint8)

    while True:
        transformed = inner_contour(inner)
        for i in range(ROW):
            for j in range(COL):
                if transformed[i, j] == 1:
                    if condition1(transformed, i, j) or condition2(transformed, i, j):
                        output[i, j] = 1
        inner = (transformed == 2).astype(np.uint8)
        if not np.any(inner):
            break
    return output


def condition1(transformed, i, j):
    ROW, COL = transformed.shape[0] - 1, transformed.shape[1] - 1
    if 0 < i < ROW:
        if (
            ((transformed[i-1, j] == 0) and (transformed[i+1, j] == 2)) or
            ((transformed[i-1, j] == 2) and (transformed[i+1, j] == 0))
        ):
            return False
    if 0 < j < COL:
        if (
            ((transformed[i, j-1] == 0) and (transformed[i, j+1] == 2)) or
            ((transformed[i, j-1] == 2) and (transformed[i, j+1] == 0))
        ):
            return False
    return True


def condition2(transformed, i, j):
    ROW, COL = transformed.shape[0] - 1, transformed.shape[1] - 1
    if (i > 0) and (j > 0):
        if (
            (transformed[i-1, j-1] == 1) and
            (transformed[i-1, j] == 0) and
            (transformed[i, j-1] == 0)
        ):
            return True
    if (i > 0) and (j < COL):
        if (
            (transformed[i-1, j+1] == 1) and
            (transformed[i-1, j] == 0) and
            (transformed[i, j+1] == 0)
        ):
            return True
    if (i < ROW) and (j > 0):
        if (
            (transformed[i+1, j-1] == 1) and
            (transformed[i+1, j] == 0) and
            (transformed[i, j-1] == 0)
        ):
            return True
    if (i < ROW) and (j < COL):
        if (
            (transformed[i+1, j+1] == 1) and
            (transformed[i+1, j] == 0) and
            (transformed[i, j+1] == 0)
        ):
            return True
    return False
