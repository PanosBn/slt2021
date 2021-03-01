import numpy as np
import math

def euclidean_distance(row1, row2): return np.sqrt(np.sum((row1 - row2)**2))

def minkowski_distance(row1, row2, p=2): 
    # p = 2: euclidian
    # p = 1: manhattan
    return (math.sum((abs(row1 - row2))**p))**(1/p)

# def accuracy_score(y_test, y_pred): return (y_test == y_pred).mean()
# I have moved accuracy score to the Knn
