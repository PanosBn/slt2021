import pandas as pandas
import numpy as np
from collections import defaultdict
from tqdm.contrib.concurrent import process_map, cpu_count, thread_map
import time
import multiprocessing
from metrics import euclidean_distance, minkowski_distance
from functools import partial
from kdTree import KdTree
from multiprocessing import Pool

num_cores = multiprocessing.cpu_count()
class KnnClassifier:
    def __init__(self, n_neighbors=5, distance_function=euclidean_distance):
        self.n_neighbors = n_neighbors
        self.tree = None
        self.distance_function = distance_function

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.tree = KdTree(self.x_train, self.y_train)


    def looc_parallel(self, X, y, return_multiple=False, tree_search=False, parallel=True):
        #implementing tqdm concurrent paradigm using functools partial
        looc_partial = partial(self.looc_helper, X=X, y=y, return_multiple=return_multiple, tree_search=tree_search)

        self.x_train = X
        self.y_train = y

        if tree_search:
            self.tree = KdTree(X, y)

        if parallel:
            max_workers = num_cores-1
        else:
            max_workers = 1

        results = process_map(looc_partial, range(len(X)), max_workers=max_workers, chunksize=max(50, int(len(X)/num_cores*2)))
        return results

    def looc_helper(self, index, X, y, return_multiple=False, tree_search=False):
        """
        :param X: matrix of features
        :param y: target label
        :returns: array of labels
        """
        if tree_search:
            pred = self.tree.search(X[index], distance_func=self.distance_function)
        else:
            pred = self.predict(X[index], return_multiple=return_multiple, single_prediction=True)
            pred = pred[0]

        return pred

    def predict(self, X, max_labels=20, return_multiple=False, single_prediction=False):
        """
        :param X: matrix of features
        :param distance_function: function to evaluate distance (euclidian, minkowski)
        :returns: array of n labels
        """
        predictions = []

        if single_prediction:
            X = [X]

        for test_digit in X:
            distances = [(self.distance_function(test_digit, digit), label) for (digit, label) in zip(self.x_train, self.y_train)]
            sorted_distances = sorted(distances, key=lambda distance: distance[0])
            k_labels = [label for (distance, label) in sorted_distances[:max_labels] if distance != 0] # so if the test X is not included
            predictions.append(k_labels)


        if return_multiple:
            return predictions
        else:
            return [max(set(pred), key=pred.count) for pred in predictions]# returns the mode / most common label

    def predict_parallel(self, X, max_labels=20, return_multiple=False, single_prediction=False):
        """
        I doubt this makes it faster as the inital splitting of 784 may take more time than the parallel processing after that - youri
        """
        predict_partial = partial(self.predict, max_labels=max_labels, return_multiple=return_multiple, single_prediction=single_prediction)
        results = process_map(predict_partial, [digit for digit in X], max_workers=num_cores-1, chunksize=max(50, int(len(X)/num_cores*2)))
        return results
    
    def loss_score(self, pred, targets, n_neighbors=None):
        assert len(pred) == len(targets), "arguments must be of same length"
        
        if not n_neighbors:
            n_neighbors = self.neighbors

        def find_label(labels):
            return max(set(labels), key=labels.count) # returns the mode / most common label
        

        pred = [find_label(array[0:n_neighbors]) for array in pred]


        loss = np.sum((pred != targets)/len(pred))

        return loss

        
    def accuracy_score(self, pred, targets, n_neighbors=None):
        """
        :param pred: list of predictions
        :param target: list of targets
        :returns: accuracy score 
        """
        assert len(pred) == len(targets), "arguments must be of same length"
        
        if not n_neighbors:
            n_neighbors = self.neighbors

        def find_label(labels):
            return max(set(labels), key=labels.count) # returns the mode / most common label
        
        pred = [find_label(array[1:1 + n_neighbors]) for array in pred]

        loss = np.mean(pred == targets)

        return loss
    

    def predict_kdtree(self, X):
        predictions = []
        for x in X:
            pred = self.tree.search(X, n_neighbors=self.n_neighbors)
            predictions.append(pred)
    
        return predictions