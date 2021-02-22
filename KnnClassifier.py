import pandas as pandas
import numpy as np
from collections import defaultdict
import time
from metrics import euclidean_distance


class KnnClassifier:
    def __init__(self, x_train, y_train, neighbours):
        self.x_train = x_train
        self.y_train = y_train
        self.neighbours = neighbours
    
    def predict(self, x_s):
        i = 0
        total_error = 0
        error = ""
        n_rows, n_col = x_s.shape
        predictions = []
        t = time.process_time()
        for test_digit in x_s:
            distances = [(euclidean_distance(test_digit, digit), label) for (digit, label) in zip(self.x_train, self.y_train)]
            sorted_distances = sorted(distances, key=lambda distance: distance[0])
            k_labels = [label for (_, label) in sorted_distances[:self.neighbours]]

            counter = defaultdict(int)
            for label in k_labels:
                counter[label] += 1
            # find the majority class:
            majority_count = max(counter.values())
            for key, value in counter.items():
                if value == majority_count:
                    pred = key
                    predictions.append(key)
 
            # output the prediction
            # i = 0
            # total_error = 0
            # error = ""
            # n_rows, n_col = x_test.shape

            # if pred != y_test[i]:
            #     total_error += 1
                # error = "ERROR"

            # print('test['+str(i)+']', '\tpredicted label is', pred, '\ttrue label is', y_test[i], error)
            i += 1
            # error = ""

        # acc = ((n_rows - total_error) / n_rows) * 100
        elapsed_time = time.process_time() - t
        print('\tFor ',self.neighbours, 'neighbours:')
        print('\tTime elapsed:', elapsed_time)
        # print('\toverall accuracy:', str(round(acc, 2))+'%')
        # print('\tnumber of errors:', total_error, 'out of', n_rows)
        return predictions

