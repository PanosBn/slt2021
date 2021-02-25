import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
#import tensorflow as tf 


def main():
    # initialize datasets from .csv files:
    train_small = pd.read_csv("data/MNIST_train_small.csv")
    test_small  = pd.read_csv("data/MNIST_test_small.csv")

    train = pd.read_csv("data/MNIST_train.csv")
    test  = pd.read_csv("data/MNIST_test_small.csv")

    '''
    # split both datasets to digits and labels (the first item in every row is a label):
    x_train = train.values[:,1:]
    y_train = train.values[:,0]
    x_test = test.values[:,1:]
    y_test = test.values[:,0]
    '''

    # or pick smaller subsets of the dataset:
    x_train = train.values[:1000,1:]
    y_train = train.values[:1000,0]
    x_test = test.values[:100,1:]
    y_test = test.values[:100,0]
    
    '''
    # show a summary
    train.info()
    test.info()
    '''

    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # plot 5 random sets of 0 - 9 digits in 5 subplots and add in one big plot
    for y, cls in enumerate(classes):
        idxs = np.nonzero([i == y for i in y_train])
        idxs = np.random.choice(idxs[0], 5)
        for i , idx in enumerate(idxs):
            plt_idx = i * len(classes) + y + 1
            plt.subplot(5, len(classes), plt_idx)
            plt.imshow(x_train[idx].reshape((28, 28)))
            plt.axis("off")
            if i == 0:
                plt.title(cls)
    
    #plt.show()

    # find euclidean distance:
    def euclidean_distance(row1, row2): return np.sqrt(np.sum((row1 - row2)**2))

    # find the most frequent label:
    def find_label(labels):
        counter = defaultdict(int)
        for label in labels:
            counter[label] += 1
        # find the majority class:
        majority_count = max(counter.values())
        for key, value in counter.items():
            if value == majority_count:
                return key
    
    # predict label:
    def predict(k, train_digits, train_labels, test_digits):
        # compute distances between the new digit and all train digits, write in list: (distance, label of train digit):
        distances = [(euclidean_distance(test_digit, digit), label) for (digit, label) in zip(train_digits, train_labels)]
        # sort the distances list by distances:
        sorted_distances = sorted(distances, key=lambda distance: distance[0])
        # extract only k closest labels:
        k_labels = [label for (_, label) in sorted_distances[:k]]
        # return the majority voted label:
        return find_label(k_labels)

    # predict for test dataset and estimate the accuracy:
    i = 0
    total_error = 0
    error = ""
    n_rows, n_col = x_test.shape
    k = 10

    for test_digit in x_test:
        pred = predict(k, x_train, y_train, x_test)
        
        # count errors:
        if pred != y_test[i]:
            total_error += 1
            error = "ERROR"

        print('test['+str(i)+']', '\tpredicted label is', pred, '\ttrue label is', y_test[i], error)
        i += 1
        error = ""

    acc = ((n_rows - total_error) / n_rows) * 100
    print('\toverall accuracy:', str(round(acc, 2))+'%')
    print('\tnumber of errors:', total_error, 'out of', n_rows)

if __name__ == "__main__":
    main()