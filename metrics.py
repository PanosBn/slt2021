import numpy as np

def euclidean_distance(row1, row2): return np.sqrt(np.sum((row1 - row2)**2))

def accuracy_score(y_test, y_pred): return (y_test == y_pred).mean()