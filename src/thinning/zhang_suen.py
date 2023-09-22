import numpy as np
from numpy.typing import NDArray


def zhang_suen(binary: NDArray[np.uint8]):
    output = np.pad(binary, 1, mode="constant", constant_values=0)
    ROW, COL = output.shape
    while True:
        change = False
        # iteration 1
        deleted = np.zeros((ROW, COL), dtype=np.bool_)
        for i in range(1, ROW-1):
            for j in range(1, COL-1):
                if output[i, j] == 1:
                    curr = output[i-1:i+2, j-1:j+2]
                    if rule1(curr) and rule2(curr) and rule3_1(curr):
                        deleted[i, j] = True
                        change = True
        output[deleted] = 0
        # iteration 2
        deleted = np.zeros((ROW, COL), dtype=np.bool_)
        for i in range(1, ROW-1):
            for j in range(1, COL-1):
                if output[i, j] == 1:
                    curr = output[i-1:i+2, j-1:j+2]
                    if rule1(curr) and rule2(curr) and rule3_2(curr):
                        deleted[i, j] = True
                        change = True
        output[deleted] = 0
        if not change:
            break
    return output[1:-1, 1:-1]


def rule1(patch):
    num_neigh = np.sum(patch) - 1
    return (2 <= num_neigh <= 6)


def rule2(patch):
    neighbors = [
        patch[0, 0], patch[0, 1], patch[0, 2],
        patch[1, 2], patch[2, 2], patch[2, 1],
        patch[2, 0], patch[1, 0], patch[0, 0]
    ]
    count = 0
    for i in range(8):
        if neighbors[i] == 0 and neighbors[i+1] == 1:
            count += 1
    return (count == 1)


def rule3_1(patch):
    trb = patch[0, 1] * patch[1, 2] * patch[2, 1]
    lbr = patch[1, 0] * patch[2, 1] * patch[1, 2]
    return (trb == 0) and (lbr == 0)


def rule3_2(patch):
    tlb = patch[0, 1] * patch[1, 0] * patch[2, 1]
    ltr = patch[1, 0] * patch[0, 1] * patch[1, 2]
    return (tlb == 0) and (ltr == 0)
