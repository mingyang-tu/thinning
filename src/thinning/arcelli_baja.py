import numpy as np
from numpy.typing import NDArray

from .distance_transform import four_distance_transform


class ArcelliBaja:
    def __init__(self, binary: NDArray[np.uint8]):
        self.binary = binary
        self.row, self.col = self.binary.shape
        self.r1 = self.row - 1
        self.c1 = self.col - 1
        self.level = 1

    def run(self):
        transformed = four_distance_transform(self.binary)
        while True:
            for i in range(self.row):
                for j in range(self.col):
                    if transformed[i, j] == self.level:
                        if not (self.condition1(transformed, i, j) or self.condition2(transformed, i, j)):
                            transformed[i, j] = 0
            if not np.any(transformed > self.level):
                break
            self.level += 1
        return (transformed > 0).astype(np.uint8)

    def condition1(self, transformed, i, j):
        if 0 < i < self.r1:
            if (
                ((transformed[i-1, j] == 0) and (transformed[i+1, j] > self.level)) or
                ((transformed[i-1, j] > self.level) and (transformed[i+1, j] == 0))
            ):
                return False
        if 0 < j < self.c1:
            if (
                ((transformed[i, j-1] == 0) and (transformed[i, j+1] > self.level)) or
                ((transformed[i, j-1] > self.level) and (transformed[i, j+1] == 0))
            ):
                return False
        return True

    def condition2(self, transformed, i, j):
        if (i > 0) and (j > 0):
            if (
                (transformed[i-1, j-1] == self.level) and
                (transformed[i-1, j] == 0) and
                (transformed[i, j-1] == 0)
            ):
                return True
        if (i > 0) and (j < self.c1):
            if (
                (transformed[i-1, j+1] == self.level) and
                (transformed[i-1, j] == 0) and
                (transformed[i, j+1] == 0)
            ):
                return True
        if (i < self.r1) and (j > 0):
            if (
                (transformed[i+1, j-1] == self.level) and
                (transformed[i+1, j] == 0) and
                (transformed[i, j-1] == 0)
            ):
                return True
        if (i < self.r1) and (j < self.c1):
            if (
                (transformed[i+1, j+1] == self.level) and
                (transformed[i+1, j] == 0) and
                (transformed[i, j+1] == 0)
            ):
                return True
        return False
