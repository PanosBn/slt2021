import pandas as pandas
import numpy as np
from collections import defaultdict
from tqdm.contrib.concurrent import process_map, cpu_count, thread_map
import time
import multiprocessing
<<<<<<< HEAD
from metrics import euclidean_distance, minkowski_distance
=======
from metrics import *
>>>>>>> youri
from functools import partial
from kdTree import KdTree
from multiprocessing import Pool
import statistics

num_cores = multiprocessing.cpu_count()
class KnnClassifier:
<<<<<<< HEAD
    def __init__(self, n_neighbors=5, distance_function=euclidean_distance):
        self.n_neighbors = n_neighbors
        self.tree = None
        self.distance_function = distance_function
=======
    def __init__(self, n_neighbors=5, distance_function="euclidean_distance"):
        distance_dict = {
            "manhattan" : manhattan,
            "euclidean_distance" : euclidean_distance,
            "canberra" : canberra,
            "chebyshev" : chebyshev,
            "braycurtis" : braycurtis,
            "cosine" : cosine,
        }

        self.n_neighbors = n_neighbors
        self.tree = None
        self.distance_function = distance_dict[distance_function]
>>>>>>> youri

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.tree = KdTree(self.x_train, self.y_train)

    def set_distance_func(self, value):
        distance_dict = {
            "manhattan" : manhattan,
            "euclidean_distance" : euclidean_distance,
            "canberra" : canberra,
            "chebyshev" : chebyshev,
            "braycurtis" : braycurtis,
            "cosine" : cosine,
        }
        self.distance_function = distance_dict[value]

<<<<<<< HEAD
    def looc_parallel(self, X, y, return_multiple=False, tree_search=False, parallel=True):
=======

    def looc_parallel(self, X, y, return_multiple=False, tree_search=False, parallel=True):
        """
        :param X: matrix of features
        :param y: target label
        :param index: necessary when called via concurrency in looc_parallel
        :param return_multiple: False: returns just the most common label and 
                                True: returns a list of n_labels labels for example [1, 1, 1, 3] when n_labels = 4
        :param parallel: whether to perform the loocv with concurrency
        :param tree_search: whether to use a kd tree search in loocv instead of the classical method
        :returns: array of labels
        """

>>>>>>> youri
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
<<<<<<< HEAD
=======
        :param index: necessary when called via concurrency in looc_parallel
        :param return_multiple: False: returns just the most common label and 
                                True: returns a list of n_labels labels for example [1, 1, 1, 3] when n_labels = 4
        :param tree_search: whether to use a kd tree search in loocv instead of the classical method
>>>>>>> youri
        :returns: array of labels
        """
        if tree_search:
            pred = self.tree.search(X[index], distance_func=self.distance_function)
        else:
<<<<<<< HEAD
            pred = self.predict(X[index], return_multiple=return_multiple, single_prediction=True)
            pred = pred[0]

        return pred

    def predict(self, X, max_labels=20, return_multiple=False, single_prediction=False):
        """
        :param X: matrix of features
        :param distance_function: function to evaluate distance (euclidian, minkowski)
=======
            pred = self.predict_helper(X[index], return_multiple=return_multiple, single_prediction=True, self_included=True)
            #pred = pred[0]

        return pred

    def predict_helper(self, X, n_labels=20, return_multiple=False, single_prediction=False, self_included=False):
        """
        :param X: list of X to predict
        :param n_labels: number of labels to return when returning multiple labels
        :param return_multiple: False: returns just the most common label and 
                                True: returns a list of n_labels labels for example [1, 1, 1, 3] when n_labels = 4
        :param single_prediction: used when being called from predict_parallel to make sure concurrency goes well
        :param self_included: whether the X to predict is included in the self.x_train and self.y_train
>>>>>>> youri
        :returns: array of n labels
        """

        predictions = []

        if single_prediction:
            X = [X]

        for test_digit in X:
            distances = [(self.distance_function(test_digit, digit), label) for (digit, label) in zip(self.x_train, self.y_train)]
            sorted_distances = sorted(distances, key=lambda distance: distance[0])
<<<<<<< HEAD
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


        loss = np.sum(pred != targets)

        return loss

        
    def accuracy_score(self, pred, targets, n_neighbors=None):
=======

            if self_included:
                k_labels = [label for (distance, label) in sorted_distances[1:n_labels+1]] # we exlcude itself (the closest) when looc is used
            else:
                k_labels = [label for (distance, label) in sorted_distances[0:n_labels]] #

            predictions.append(k_labels)

        if return_multiple:
            return predictions[0]
        else:
            return [statistics.multimode(pred[:self.n_neighbors])[0] for pred in predictions][0] # returns the mode / most common label

    def predict_parallel(self, X, n_labels=20, return_multiple=False, self_included=False, tree_search=False):
        """
        :param X: list of X to predict
        :param n_labels: number of labels to return when returning multiple labels
        :param return_multiple: False: returns just the most common label and 
                                True: returns a list of n_labels labels for example [1, 1, 1, 3] when n_labels = 4
        :param self_included: whether the X to predict is included in the self.x_train and self.y_train
        :returns: parallel computed predict() results in a list of neighboring labels or single value of most common neighboring labels
        """
        if tree_search:
            predict_partial = partial(self.predict_kdtree, n_labels=n_labels, return_multiple=return_multiple, single_prediction=True, self_included=self_included)
        else:
            predict_partial = partial(self.predict_helper, n_labels=n_labels, return_multiple=return_multiple, single_prediction=True, self_included=self_included)
        results = process_map(predict_partial, X, max_workers=num_cores-1, chunksize=max(50, int(len(X)/num_cores*2)))
        return results
    
    def score(self, pred, targets, how="loss", n_neighbors=None, multiple=False):
        """
        :param pred: list of predictions
        :param target: list of targets
        :param how: string of way to score such as loss, risk, accuracy
        :param n_neighbors: number of neighbors to consider when using scoring for an array of multiple labels such as [1,1,2,1]
        :param multiple: whether input is a single label or an array of labels
        :returns: loss score 
        """
        assert len(pred) == len(targets), "arguments must be of same length"
        
        if not n_neighbors:
            n_neighbors = self.n_neighbors

        
        if multiple:
        # statistics.multimode returns multi modes (if exists) 
        # and chooses the first it passes as the first it returns in the list of modes
        # this solves ties and gives privilege to the closer neighbors
            pred = [statistics.multimode(array[0:n_neighbors])[0] for array in pred]
        
        if how == "loss":
            result = np.mean(pred != targets)
        elif how == "accuracy":
            result = np.mean(pred == targets)
        elif how == "risk":
            print("still needs to be implemented")

        return result

    def predict_kdtree(self, X, n_labels=20, return_multiple=False, self_included=False, single_prediction=False):
>>>>>>> youri
        """
        :param X: list of X to predict
        :param n_labels: number of labels to return when returning multiple labels
        :param return_multiple: False: returns just the most common label and 
                                True: returns a list of n_labels labels for example [1, 1, 1, 3] when n_labels = 4
        :param single_prediction: used when being called from predict_parallel to make sure concurrency goes well
        :param self_included: whether the X to predict is included in the self.x_train and self.y_train
        :returns: array of n labels
        """
<<<<<<< HEAD
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
=======
        predictions = []

        if single_prediction:
            X = [X]

        for test_digit in X:
            pred = self.tree.search(test_digit, n_labels=n_labels)

            if return_multiple:
                if self_included:
                    pred = pred[1:1 + n_labels] # we exlcude itself (the closest) when self is included is used
                else:
                    pred = pred[:n_labels]
            else:
                if self_included:
                    pred = statistics.multimode(pred[1:self.n_neighbors])[0] # we exlcude itself (the closest) when self is included is used
                else:
                    pred = statistics.multimode(pred[:self.n_neighbors])[0]
            
            predictions.append(pred)

        if single_prediction:
            return predictions[0]
        else:
            return predictions
>>>>>>> youri
