import numpy as np
import scipy as sp 
from scipy.spatial import distance

def euclidean_distance(row1, row2): return np.sqrt(np.sum((row1 - row2)**2))

def minkowski_distance(row1, row2, p=2): 
    # p = 2: euclidian
    # p = 1: manhattan
    return (np.sum((np.float128(abs(row1 - row2)))**p))**(1/p)

def manhattan(row1,row2): return distance.cityblock(row1, row2)

def canberra(row1, row2): return distance.canberra(row1, row2)

def chebyshev(row1 ,row2): return distance.chebyshev(row1,row2)

def braycurtis(row1, row2): return distance.braycurtis(row1, row2)

def cosine(row1 ,row2): return distance.cosine(row1,row2)

def hamming(row1, row2) : return distance.hamming(row1, row2)

