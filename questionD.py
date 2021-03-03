import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
import imblearn
from datetime import datetime 
from collections import defaultdict
from metrics import *
from KnnClassifier import KnnClassifier
from tqdm import tqdm
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

def plot_q3(supersampled, normal, l):
    x_range = np.linspace(1, l, num=l)
    fig = plt.figure()
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(x_range, supersampled, label='Supersampled', marker='.')
    ax1.plot(x_range, normal, label='Normal', marker='.')
    ax1.set_xlabel("Number of neighbours")
    ax1.set_ylabel("Loss")
    ax1.set_title("Test loss")
    ax1.legend()
    
    plt.show()


def plot_question_D(dict, l):
    x_range = np.linspace(1, l, num=l)
    fig = plt.figure()
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(x_range, supersampled, label='Supersampled', marker='.')
    ax1.plot(x_range, normal, label='Normal', marker='.')
    ax1.set_xlabel("Number of neighbours")
    ax1.set_ylabel("Loss")
    ax1.set_title("Test loss")
    ax1.legend()
    
    plt.show()

def main():
    # initialize datasets from .csv files:
    train_small = pd.read_csv("data/MNIST_train_small.csv", nrows=1500)
    test_small  = pd.read_csv("data/MNIST_test_small.csv", nrows=500)

    train_large = pd.read_csv("data/MNIST_train.csv")

    # split both datasets to digits and labels (the first item in every row is a label):
    x_train = train_small.values[:,1:]
    y_train = train_small.values[:,0]

    x_test = test_small.values[:,1:]
    y_test = test_small.values[:,0]

    print("X_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)
 
    x_train_large = train_large.values[:,1:]/255
    y_train_large = train_large.values[:,0]

    print(type(x_train_large))

    print("X_train large shape: ", x_train.shape)
    print("y_train large shape: ", y_train.shape)

    print("Class counts for the normal dataset")
    y_train_df = pd.DataFrame(data = y_train, columns=["class"])
    print(y_train_df['class'].value_counts())

    # _ = y_train_df['class'].value_counts().plot(kind='bar')
    # plt.show()
    
    oversample = SMOTE()
    X_super, y_super = oversample.fit_resample(x_train,y_train)

    print("Class counts for the supersampled dataset")
    y_supersampled_df = pd.DataFrame(data = y_super, columns=["class"])
    print(y_supersampled_df['class'].value_counts())

    test_predictions, train_predictions = [], []
    looc_test_predictions, looc_train_predictions = [], []
    supersampled_predictions = {}

    distance_functions = ['cosine','braycurtis','sqeuclidean']
    ll = 2
    # loop over number of neighbors
    start_time = datetime.now() 
    for distance in distance_functions:
        clf = KnnClassifier(n_neighbors=5)
        clf_supersampled = KnnClassifier(n_neighbors=5)
        clf.fit(x_train, y_train)
        clf_supersampled.fit(X_super,y_super)

        # normal parallel
        accuracy_train_supersampled = clf_supersampled.accuracy_score(clf_supersampled.predict_for_question_D(x_test,distance), y_test)
        # accuracy_train = clf.accuracy_score(clf.predict_parallel(x_test,distance,2), y_test)
        # supersampled_predictions.append(accuracy_train_supersampled)
        # train_predictions.append(accuracy_train)
        supersampled_predictions.append({"Distance Function" : distance, "Loss": accuracy_train_supersampled})
        print(supersampled_predictions)

    time_elapsed = datetime.now() - start_time 

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    # plot_question_D(supersampled_predictions)
    

if __name__ == "__main__":
    main()