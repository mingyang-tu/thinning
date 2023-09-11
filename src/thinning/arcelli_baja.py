import numpy as np

from .distance_transform import inner_contour


class ArcelliBaja:
    def __init__(self, binary):     # input : np.uint8
        self.binary = binary
        self.row, self.col = self.binary.shape
        self.r1 = self.row - 1
        self.c1 = self.col - 1

    # FIXME: isolated point
    def run(self):
        inner = self.binary.copy()
        output = np.zeros((self.row, self.col), dtype=np.uint8)

        while True:
            transformed = inner_contour(inner)
            for i in range(self.row):
                for j in range(self.col):
                    if transformed[i, j] == 1:
                        if self.condition1(transformed, i, j) or self.condition2(transformed, i, j):
                            output[i, j] = 1
            inner = (transformed == 2).astype(np.uint8)
            if not np.any(inner):
                break
        return output

    def condition1(self, transformed, i, j):
        if 0 < i < self.r1:
            if (
                ((transformed[i-1, j] == 0) and (transformed[i+1, j] == 2)) or
                ((transformed[i-1, j] == 2) and (transformed[i+1, j] == 0))
            ):
                return False
        if 0 < j < self.c1:
            if (
                ((transformed[i, j-1] == 0) and (transformed[i, j+1] == 2)) or
                ((transformed[i, j-1] == 2) and (transformed[i, j+1] == 0))
            ):
                return False
        return True

    def condition2(self, transformed, i, j):
        if (i > 0) and (j > 0):
            if (
                (transformed[i-1, j-1] == 1) and
                (transformed[i-1, j] == 0) and
                (transformed[i, j-1] == 0)
            ):
                return True
        if (i > 0) and (j < self.c1):
            if (
                (transformed[i-1, j+1] == 1) and
                (transformed[i-1, j] == 0) and
                (transformed[i, j+1] == 0)
            ):
                return True
        if (i < self.r1) and (j > 0):
            if (
                (transformed[i+1, j-1] == 1) and
                (transformed[i+1, j] == 0) and
                (transformed[i, j-1] == 0)
            ):
                return True
        if (i < self.r1) and (j < self.c1):
            if (
                (transformed[i+1, j+1] == 1) and
                (transformed[i+1, j] == 0) and
                (transformed[i, j+1] == 0)
            ):
                return True
        return False
