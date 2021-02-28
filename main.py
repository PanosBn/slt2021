import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime 
from collections import defaultdict
from metrics import euclidean_distance, minkowski_distance
from KnnClassifier import KnnClassifier
from tqdm import tqdm
import sys
import time



def plot_q3(y_train_predictions, y_test_predictions, l):
    x_range = np.linspace(1, l, num=l)
    fig = plt.figure()
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(x_range, y_train_predictions, label='Training set', marker='.')
    ax1.plot(x_range, y_test_predictions, label='Test set', marker='.')
    ax1.set_xlabel("Number of neighbours")
    ax1.set_ylabel("Accuracy (%))")
    ax1.set_title("Accuracy")
    ax1.legend()
    
    # ax2.plot(x_range, grd_performance, label='Gradient Descent')
    # ax2.plot(x_range, quad_performance, label='Quadratic')
    # ax2.set_xlabel("Dataset size (%)")
    # ax2.set_ylabel("Testing accuracy (%)")
    # ax2.set_title("Testing Accuracy")
    # ax2.legend()
    plt.show()


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
    

    # or pick smaller subsets of the dataset:
    #x_train = train.values[:2900,1:]
    #y_train = train.values[:2900,0]

    # clf = KnnClassifier(x_train,y_train,5).predict(x_test,y_test)
    # k = 5
    '''
    y_train_predictions_set = []
    y_test_predictions_set= []
    y_test_prediction_set_errors = []
    y_train_predictions_set_errors= []
    y_train_loss = 0
    y_test_loss = 0
    
    '''
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



    ll = 11 
    # loop over number of neighbors
    start_time = datetime.now() 
    for k in tqdm(range(1,ll)):
        clf = KnnClassifier(n_neighbors=k)
        clf.fit(x_train, y_train)

        # # normal
        # accuracy_test = clf.accuracy_score(clf.predict(x_test), y_test)
        # accuracy_train = clf.accuracy_score(clf.predict(x_train), y_train)
        # test_predictions.append(accuracy_test)
        # train_predictions.append(accuracy_train)

        # normal parallel
        # accuracy_test = clf.accuracy_score(clf.predict_parallel(x_test), y_test)
        # accuracy_train = clf.accuracy_score(clf.predict_parallel(x_train), y_train)
        # test_predictions.append(accuracy_test)
        # train_predictions.append(accuracy_train)

        # looc
        # looc_accuracy_train = clf.looc_validate(x_train, y_train)
        # looc_accuracy_test = clf.looc_validate(x_test, y_test)
        # looc_test_predictions.append(looc_accuracy_test)
        # looc_train_predictions.append(looc_accuracy_train)

        # looc parallel tree
        """
        t = time.time()
        tree_single_score.append(clf.looc_single_kdtree(x_train, y_train))
        tree_single_time.append(time.time() - t)
        """
        t = time.time()
        tree_parallel_score.append(clf.looc_parallel_kdtree(x_train, y_train))
        tree_parallel_time.append(time.time() - t)
        """
        t = time.time()
        looc_parallel_score.append(clf.looc_validate_parallel(x_train, y_train))
        looc_parallel_time.append(time.time() - t)
        """
    fig, axs = plt.subplots(2)
    axs[0].set_title("score vs neighbors")
    #axs[0].plot(range(1,ll), looc_parallel_score, label="parallel")
    axs[0].plot(range(1,ll), tree_parallel_score, label="parallel tree")
    #axs[0].plot(range(1,ll), tree_single_score, label="single tree", linestyle="-.")
    axs[0].set_xlabel("neighbors")
    axs[0].set_ylabel("score")
    axs[0].legend()

    axs[1].set_title("time vs neighbors")
    axs[1].set_xlabel("neighbors")
    axs[1].set_ylabel("time")
    #axs[1].plot(range(1,ll), looc_parallel_time, label="parallel")
    axs[1].plot(range(1,ll), tree_parallel_time, label="parallel tree")
    #axs[1].plot(range(1,ll), tree_single_time, label="single tree", linestyle="-.")
    axs[1].legend()
    plt.show()

    time_elapsed = datetime.now() - start_time 

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    


    # plot graphs train vs test score vs n neighbors
    # plot_q3(train_predictions, test_predictions, ll-1)
    plot_q3(tree_train_predictions, tree_test_predictions, ll-1)

    


    # y_train_predictions_set = y_train_predictions_set.sort()

    # y_test_predictions_set = y_test_predictions_set.sort()
    # df_result = pd.DataFrame()
    # df_result['K value'] = [1,2,3]
    # df_result['train pred'] = y_train_predictions_set
    # df_result['test pred'] = y_test_predictions_set
    # print(df_result)

    # print(len(y_train_predictions))
    # # print(type(y_train[:10]))
    # print(type(y_train_predictions))
    # train_accuracy = accuracy_score(y_train[:10], y_test_predictions[:10])
    # print("Train accuracy: ", train_accuracy)
    # print(y_train[:10])
    # print(y_train_predictions)
    
    # classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # # # plot 5 random sets of 0 - 9 digits in 5 subplots and add in one big plot
    # # for y, cls in enumerate(classes):
    # #     idxs = np.nonzero([i == y for i in y_train])
    # #     idxs = np.random.choice(idxs[0], 5)
    # #     for i , idx in enumerate(idxs):
    # #         plt_idx = i * len(classes) + y + 1
    # #         plt.subplot(5, len(classes), plt_idx)
    # #         plt.imshow(x_train[idx].reshape((28, 28)))
    # #         plt.axis("off")
    # #         if i == 0:
    # #             plt.title(cls)
    
    # # #plt.show()


if __name__ == "__main__":
    main()