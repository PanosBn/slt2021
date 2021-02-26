import numpy as np


"""
:Types of distances 
:minkowski distance is the general form of distance. When p = 1, we get the manhattan distance
:When p = 2 we get the euclidean distance
:When p -> inf we get the chebyshev distance

:The weighted minkowski distance calculates the distance between two vectors w.r.t. the weight of these points
:
:returns: 
"""

def euclidean_distance(row1, row2): return np.sqrt(np.sum((row1 - row2)**2))

def minkowski_distance(row1, row2, p): return (np.sum(abs((row1 - row2))**p))**(1/p)

def weighted_minkowski_distance(row1, row2, p, weight): return (np.sum(abs(weight*(row1-row2))**p))**(1/p)

# def accuracy_score(y_test, y_pred): return (y_test == y_pred).mean()
# I have moved accuracy score to the Knn
