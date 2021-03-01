import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime 
from collections import defaultdict
import metrics
from KnnClassifier import KnnClassifier
from tqdm import tqdm
import sys
import time



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

            kp_accuracy_train = clf.looc_validate_parallel(x_train, y_train, metrics.minkowski_distance, p)
            kp_accuracy_test = clf.looc_validate_parallel(x_test, y_test, metrics.minkowski_distance, p)
            kp_train_predictions[k-1, p-1] = kp_accuracy_train
            kp_test_predictions[k-1, p-1] = kp_accuracy_test

    print(kp_train_predictions, "\n\n", kp_test_predictions)
    plot_kp(kp_train_predictions, kp_test_predictions, max_k)


def main():
    sys.setrecursionlimit(10000)

    # initialize datasets from .csv files:
    train_small = pd.read_csv("data/MNIST_train_small.csv", nrows=1000, header=None)
    test_small  = pd.read_csv("data/MNIST_test_small.csv", nrows=500, header=None)

    
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
    tree_test_predictions, tree_train_predictions = [], []
    tree_single_score, tree_parallel_score, looc_parallel_score  = [], [], []
    tree_single_time, tree_parallel_time, looc_parallel_time  = [], [], []
    
    def plot_trainspeed(X, y):
        tree_single_score, tree_parallel_score, looc_parallel_score, looc_single_score  = [], [], [], []
        tree_single_time, tree_parallel_time, looc_parallel_time, looc_single_time  = [], [], [], []
        size = 110

        # looc with classic NN single processing
        print("scoring looc with classic NN single processing...")
        for i in tqdm(range(10, size, 10)):
            X_ = X[:round(len(X)*0.01*size)]
            y_ = y[:round(len(y)*0.01*size)]
            clf = KnnClassifier(n_neighbors=1)
            clf.fit(X_, y_)
            
            t = time.time()
            looc_single_score.append(clf.looc_single_kdtree(X_, y_))
            looc_single_time.append(time.time() - t)
        
        # looc with classic NN multi processing
        print("scoring looc with classic NN multi processing...")
        for i in tqdm(range(10, size, 10)):
            X_ = X[:round(len(X)*0.01*size)]
            y_ = y[:round(len(y)*0.01*size)]
            clf = KnnClassifier(n_neighbors=2)
            clf.fit(X_, y_)

            t = time.time()
            looc_parallel_score.append(clf.looc_validate_parallel(X_, y_))
            looc_parallel_time.append(time.time() - t)
        
        # looc with kd tree single processing
        print("scoring looc with kd tree NN single processing...")
        for i in tqdm(range(10, size, 10)):
            X_ = X[:round(len(X)*0.01*size)]
            y_ = y[:round(len(y)*0.01*size)]
            clf = KnnClassifier(n_neighbors=1)
            
            t = time.time()
            tree_single_score.append(clf.looc_single_kdtree(X_, y_))
            tree_single_time.append(time.time() - t)

        # looc with kd tree multi processing
        print("scoring looc with kd tree NN multi processing...")
        for i in tqdm(range(10, size, 10)):
            X_ = X[:round(len(X)*0.01*size)]
            y_ = y[:round(len(y)*0.01*size)]
            clf = KnnClassifier(n_neighbors=1)        
            
            t = time.time()
            tree_parallel_score.append(clf.looc_parallel_kdtree(X_, y_))
            tree_parallel_time.append(time.time() - t)
            
        fig, axs = plt.subplots(2)
        axs[0].set_title("score vs neighbors")
        axs[0].plot(range(10, size, 10), looc_single_score, label="classic single", linestyle="-")
        axs[0].plot(range(10, size, 10), looc_parallel_score, label="classic parallel", linestyle="-.")
        axs[0].plot(range(10, size, 10), tree_single_score, label="tree single", linestyle="-")
        axs[0].plot(range(10, size, 10), tree_parallel_score, label="tree parallel", linestyle="-.")
        axs[0].set_xlabel("traindata set size in %")
        axs[0].set_ylabel("score")
        axs[0].set_xticks(range(10, size, 10))
        axs[0].legend()

        axs[1].set_title("time vs neighbors")
        axs[1].set_xlabel("traindata set size in %")
        axs[1].set_ylabel("time")
        axs[1].plot(range(10, size, 10), looc_single_time, label="classic single", linestyle="-.")
        axs[1].plot(range(10, size, 10), looc_parallel_time, label="classic parallel", linestyle="-.")
        axs[1].plot(range(10, size, 10), tree_single_time, label="tree single", linestyle="-.")
        axs[1].plot(range(10, size, 10), tree_parallel_time, label="tree parallel", linestyle="-.")
        axs[1].set_xticks(range(10, size, 10))
        axs[1].legend()

        plt.tight_layout()
        plt.show()


    plot_trainspeed(x_train, y_train)



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