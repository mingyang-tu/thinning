import numpy as np

from .distance_transform import inner_contour


def arcelli_baja(binary):       # input : np.uint8
    ROW, COL = binary.shape
    inner = binary.copy()
    output = np.zeros((ROW, COL), dtype=np.uint8)

    while True:
        transformed = inner_contour(inner)
        for i in range(ROW):
            for j in range(COL):
                if transformed[i, j] == 1:
                    # TODO: condition1 and condition2
                    pass


def condition1(transformed, i, j):
    ROW, COL = transformed.shape
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
    # TODO: condition2
    pass