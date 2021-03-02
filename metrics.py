import numpy as np


def euclidean_distance(row1, row2): return np.sqrt(np.sum((row1 - row2)**2))

def minkowski_distance(row1, row2, p=2): 
    # p = 2: euclidian
    # p = 1: manhattan
    return (np.sum((np.uint64(abs(row1 - row2)))**p))**(1/p)

