import pandas as pandas
import numpy as np
from collections import defaultdict
from tqdm.contrib.concurrent import process_map, cpu_count, thread_map
import time
import multiprocessing
import metrics
from metrics import *
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

        results = process_map(looc_partial, range(len(X)), max_workers=max_workers, chunksize=200)
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
            pred = self.predict(X[index], return_multiple=return_multiple, single_prediction=True, self_included=True)
            pred = pred[0]

        return pred

    def predict(self, X, max_labels=20, return_multiple=False, single_prediction=False, self_included=False):
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

            if self_included:
                k_labels = [label for (distance, label) in sorted_distances[1:max_labels+1]] # we exlcude itself (the closest) when looc is used
            else:
                k_labels = [label for (distance, label) in sorted_distances[:max_labels]] #

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
        results = process_map(predict_partial, [digit for digit in X], max_workers=num_cores-1, chunksize=200)
        return results

    def predict_parallel_old(self,X, distance_function):
        predict_partial = partial(self.predict_single, distance_function=distance_function)
        results = process_map(predict_partial, [digit for digit in X], max_workers=num_cores-1, chunksize=200)
        # print(results)
        # predictions, targets = zip(*results)

        return results
    

    def predict_single(self, test_digit, distance_function):

        myDict = {
            "euclidean_distance" : euclidean_distance,
            "canberra" : canberra,
            "chebyshev" : chebyshev,
            "braycurtis" : braycurtis,
            "cosine" : cosine,
            "seuclidean" : seuclidean,
        }

        # if (distance_function == 'euclidean'):
        #     dist_func = euclidean_distance
        # switch (distance_function) :
        #     case 1:
        def find_label(labels):
            return max(set(labels), key=labels.count) # returns the mode / most common label

        distances = [(myDict[distance_function](test_digit, digit), label) for (digit, label) in zip(self.x_train, self.y_train)]
        sorted_distances = sorted(distances, key=lambda distance: distance[0])
        k_labels = [label for (_, label) in sorted_distances[:self.n_neighbors]]
        prediction = (find_label(k_labels))

        return prediction
    
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
            n_neighbors = self.n_neighbors

        def find_label(labels):
            return max(set(labels), key=labels.count) # returns the mode / most common label
        
        pred = [find_label(array[0:n_neighbors]) for array in pred]

        loss = np.mean(pred == targets)

        return loss
    def accuracy_score_old(self, pred, target):
        """
        :param pred: list of predictions
        :param target: list of targets
        :returns: accuracy score 
        """

        assert len(pred) == len(target), "arguments must be of same length"

        return np.mean(1-np.equal(pred, target, dtype=int))
    

    def predict_kdtree(self, X):
        predictions = []
        for x in X:
            pred = self.tree.search(X, n_neighbors=self.n_neighbors)
            predictions.append(pred)
    
        return predictions