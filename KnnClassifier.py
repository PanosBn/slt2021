import pandas as pandas
import numpy as np
from collections import defaultdict
from tqdm.contrib.concurrent import process_map, cpu_count
import time
import multiprocessing
from metrics import euclidean_distance,minkowski_distance
from functools import partial

num_cores = multiprocessing.cpu_count()
class KnnClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def looc_validate_parallel(self, X, y):
        #implementing tqdm concurrent paradigm using functools partial
        looc_partial = partial(self.looc_single, X=X, y=y)
        results = process_map(looc_partial, range(len(X)), max_workers=num_cores, chunksize=max(50, int(len(X)/num_cores*2)))
        predictions, targets = zip(*results)

        return self.accuracy_score(predictions, targets)
        

    def looc_single(self, index, X, y):

        self.x_train = X
        self.y_train = y
        np.delete(self.x_train, index)
        np.delete(self.y_train, index)

        pred = self.predict_single(X[index])
        # print("PRINTING PREDICTION: ", pred)
        target = y[index]
        # print("printing Target Label: ", target)

        return pred, target


    def looc_validate(self, X, y):
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
            np.delete(self.x_train, i)
            np.delete(self.y_train, i)

            pred = self.predict([X[i]]) # has to take a list
            predictions.append(*pred) # returns a list therefore unpack with *
            targets.append(y[i])        

        return self.accuracy_score(predictions, targets) 

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
        print(results)
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

        return np.mean(1-np.equal(pred, target, dtype=int))

        # incorrect, correct = 0, 0

        # for predicted, actual in zip(pred, target):
        #     if predicted != actual: 
        #         incorrect += 1
        #     else:
        #         correct += 1
        #     #print(f"predicted: {predicted}, actual: {actual}, correct: {correct}, incorrect:{incorrect}")

        # return correct / (correct + incorrect)