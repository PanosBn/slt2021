import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime 
from collections import defaultdict
import metrics
from KnnClassifier import KnnClassifier
from tqdm import tqdm


def plot_accuracy(y_train_basic, y_test_basic, y_train_loocv, y_test_loocv, max_k):
    x_range = np.linspace(1, max_k, num=max_k)
    fig = plt.figure()
    fig.set_figwidth(15)
    
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x_range, y_train_basic, label='Training set, basic', marker='.')
    ax1.plot(x_range, y_test_basic, label='Test set, basic', marker='.')
    ax1.plot(x_range, y_train_loocv, label='Training set, LOOC', marker='.')
    ax1.plot(x_range, y_test_loocv, label='Test set, LOOC', marker='.')
    ax1.set_xlabel("Number of neighbours")
    ax1.set_ylabel("Accuracy (%))")
    ax1.set_title("Accuracy basic vs. LOOC")
    ax1.legend()

    plt.show()
    print("test")

def plot_kp(train, test, max_k):
    x_range = np.linspace(1, max_k, num=max_k)
    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(20)
    _, ncol = train.shape
    axs = {}

    for i in range(0, ncol):
        axs[i] = fig.add_subplot(5, 3, i+1)
        
        axs[i].plot(x_range, train[:,i], label=f'Training set p = {i+1}', marker='.')
        axs[i].plot(x_range, test[:,i], label=f'Test set p = {i+1}', marker='.')

        axs[i].set_xlabel("Number of neighbours")
        axs[i].set_ylabel("Accuracy (%))")
        axs[i].set_title("Accuracy using k, p")
        axs[i].legend()
        
    plt.show()

def compare_over_kp(x_train, y_train, x_test, y_test, max_k, max_p):
    kp_test_predictions = np.zeros((max_k, max_p))
    kp_train_predictions = np.zeros((max_k, max_p))

    for k in tqdm(range(1,max_k+1)):
        for p in range(1, max_p+1):
            clf = KnnClassifier(n_neighbors=k)
            clf.fit(x_train, y_train)

            kp_accuracy_train = clf.looc_validate(x_train, y_train, metrics.minkowski_distance, p)
            kp_accuracy_test = clf.looc_validate(x_test, y_test, metrics.minkowski_distance, p)
            kp_train_predictions[k-1, p-1] = kp_accuracy_train
            kp_test_predictions[k-1, p-1] = kp_accuracy_test

    print(kp_train_predictions, "\n\n", kp_test_predictions)
    plot_kp(kp_train_predictions, kp_test_predictions, max_k)


def main():
    # initialize datasets from .csv files:
    nr_rows = 40
    train_small = pd.read_csv("data/MNIST_train_small.csv", nrows=nr_rows)
    test_small  = pd.read_csv("data/MNIST_test_small.csv", nrows=nr_rows)
    
    # split both datasets to digits and labels (the first item in every row is a label):
    x_train = train_small.values[:,1:]
    y_train = train_small.values[:,0]
    x_test = test_small.values[:,1:]
    y_test = test_small.values[:,0]
    
    # max number of neighbors
    max_k = 4

    # max degree for minkowski distance
    max_p = 4

    # plot graphs for question C
    compare_over_kp(x_train, y_train, x_test, y_test, max_k, max_p)

    test_predictions, train_predictions = [], []
    looc_test_predictions, looc_train_predictions = [], []

    # loop over number of neighbors
    start_time = datetime.now() 
    for k in tqdm(range(1,max_k+1)):
        clf = KnnClassifier(n_neighbors=k)
        clf.fit(x_train, y_train)

        # normal
        accuracy_train = clf.accuracy_score(clf.predict(x_train), y_train)
        accuracy_test = clf.accuracy_score(clf.predict(x_test), y_test)
        train_predictions.append(accuracy_train)
        test_predictions.append(accuracy_test)

        # # looc
        # looc_accuracy_train = clf.looc_validate(x_train, y_train)
        # looc_accuracy_test = clf.looc_validate(x_test, y_test)
        # looc_train_predictions.append(looc_accuracy_train)
        # looc_test_predictions.append(looc_accuracy_test)

        looc_accuracy_train = clf.looc_validate_parallel(x_train, y_train)
        looc_accuracy_test = clf.looc_validate_parallel(x_test, y_test)
        looc_test_predictions.append(looc_accuracy_test)
        looc_train_predictions.append(looc_accuracy_train)

    time_elapsed = datetime.now() - start_time 

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    print(looc_train_predictions)

    # plot graphs train vs test score vs n neighbors
    plot_accuracy(train_predictions, test_predictions, looc_train_predictions, looc_test_predictions, max_k)

if __name__ == "__main__":
    main()