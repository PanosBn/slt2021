import pandas as pandas
import numpy as np
from collections import defaultdict
from tqdm.contrib.concurrent import process_map, cpu_count, thread_map
import time
import multiprocessing
from metrics import euclidean_distance,minkowski_distance
from functools import partial
from kdTree import KdTree
from multiprocessing import Pool

num_cores = multiprocessing.cpu_count()
class KnnClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.tree = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.tree = KdTree(self.x_train, self.y_train)


    def looc_validate_parallel(self, X, y, distance_function=euclidean_distance):
        #implementing tqdm concurrent paradigm using functools partial
        looc_partial = partial(self.looc_single, X=X, y=y)
        results = process_map(looc_partial, range(len(X)), max_workers=num_cores-1, chunksize=max(50, int(len(X)/num_cores*2)))
        predictions, targets = zip(*results)

        return self.accuracy_score(predictions, targets)

    def looc_single_kdtree(self, X, y, tree=None, distance_function=euclidean_distance):
        predictions, targets = [], y
        self.tree = KdTree(X, y)

        for x in X:
            pred = self.tree.search(x, n_neighbors=self.n_neighbors, self_included=True)
            predictions.append(pred)
        
        return self.accuracy_score(predictions, targets)

    def looc_parallel_kdtree(self, X, y, distance_function=euclidean_distance):
        pool = Pool(maxtasksperchild=50, processes=num_cores-1)
        self.tree = KdTree(X,y)
        part = partial(self.tree.search, n_neighbors=self.n_neighbors, self_included=True)
        predictions = process_map(part, X, max_workers=num_cores-1, chunksize=max(50, int(len(X)/num_cores*2)))
        
        return self.accuracy_score(predictions, y)
        
    def predict_kdtree(self, X, distance_function=euclidean_distance):
        predictions = []
        for x in X:
            pred = self.tree.search(X, n_neighbors=self.n_neighbors)
            predictions.append(pred)
    
        return predictions

    def looc_single(self, index, X, y, distance_function=euclidean_distance):

        self.x_train = X
        self.y_train = y
        self.x_train = np.delete(self.x_train, index)
        self.y_train = np.delete(self.y_train, index)

        pred = self.predict_single(X[index])
        # print("PRINTING PREDICTION: ", pred)
        target = y[index]
        # print("printing Target Label: ", target)

        return pred, target


    def looc_validate(self, X, y, distance_function=euclidean_distance):
        """
        :param X: matrix of features
        :param y: target label
        :returns: total leave-one-out-cross-validation score
        """
        predictions, targets = [], []
        for i in range(0, len(y)): # we loop over each instance and subtract it from the train set (looc)
            
            # effictively the same as fitting just minus one instance
            self.x_train = X
            self.y_train = y
            self.x_train = np.delete(self.x_train, i)
            self.y_train = np.delete(self.y_train, i)

            pred = self.predict([X[i]], distance_function=euclidean_distance) # has to take a list
            predictions.append(*pred) # returns a list therefore unpack with *
            targets.append(y[i])        

        return self.accuracy_score(predictions, targets) 

    def find_kp(self, X, y, distance_function=minkowski_distance):
        """
        Loops over k and p and computes the accuracy 
        for the combinations of k, p values. 
        """
        stats = []
        for k in range(0,15):
            for p in range(0,10):
                clf = KnnClassifier(n_neighbors=k)
                clf.fit(X, y)

                stats.append(k, p, clf.looc_validate(X, y, disance_function=minkowski_distance))
        return stats

    def predict(self, X, distance_function=euclidean_distance):
        """
        :param X: matrix of features
        :param distance_function: function to evaluate distance (euclidian, minkowski)
        :returns: predictions in list
        """
        predictions = []
        t = time.process_time()

        def find_label(labels):
            return max(set(labels), key=labels.count) # returns the mode / most common label

        for test_digit in X:
            distances = [(distance_function(test_digit, digit), label) for (digit, label) in zip(self.x_train, self.y_train)]
            sorted_distances = sorted(distances, key=lambda distance: distance[0])
            k_labels = [label for (_, label) in sorted_distances[:self.n_neighbors]]
            predictions.append(find_label(k_labels))

        elapsed_time = time.process_time() - t

        #print('\tFor ',self.n_neighbors, 'neighbours:')
        #print('\tTime elapsed:', elapsed_time)
        return predictions

    def predict_parallel(self,X, distance_function=euclidean_distance):
        predict_partial = partial(self.predict_single)
        results = process_map(predict_partial, [digit for digit in X], max_workers=num_cores-1, chunksize=max(50, int(len(X)/num_cores*2)))
        #print(results)
        # predictions, targets = zip(*results)

        return results

    def predict_single(self, test_digit, distance_function=euclidean_distance):

        def find_label(labels):
            return max(set(labels), key=labels.count) # returns the mode / most common label

        distances = [(distance_function(test_digit, digit), label) for (digit, label) in zip(self.x_train, self.y_train)]
        sorted_distances = sorted(distances, key=lambda distance: distance[0])
        k_labels = [label for (_, label) in sorted_distances[:self.n_neighbors]]
        prediction = (find_label(k_labels))

        return prediction
        
    def accuracy_score(self, pred, target):
        """
        :param pred: list of predictions
        :param target: list of targets
        :returns: accuracy score 
        """
        assert len(pred) == len(target), "arguments must be of same length"

        incorrect, correct = 0, 0

        for predicted, actual in zip(pred, target):
            if predicted != actual: 
                incorrect += 1
            else:
                correct += 1
            #print(f"predicted: {predicted}, actual: {actual}, correct: {correct}, incorrect:{incorrect}")

        return correct / (correct + incorrect)