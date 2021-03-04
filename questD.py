import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import seaborn as sns
from datetime import datetime 
from collections import defaultdict
from KnnClassifier import KnnClassifier
from tqdm import tqdm
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


def main():
    # initialize datasets from .csv files:
    train_small = pd.read_csv("data/MNIST_train_small.csv", nrows=3000, header=None)
    test_small  = pd.read_csv("data/MNIST_test_small.csv", nrows=500, header=None)

    # split both datasets to digits and labels (the first item in every row is a label):
    x_train = train_small.values[:,1:]
    y_train = train_small.values[:,0]
    x_test = test_small.values[:,1:]
    y_test = test_small.values[:,0]

    print("X_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)


    # print("Class counts for the normal dataset")
    # y_train_df = pd.DataFrame(data = y_train, columns=["class"])
    # # print(y_train_df['class'].value_counts())

    # # _ = y_train_df['class'].value_counts().plot(kind='bar')
    # # plt.show()
    
    oversample = SMOTE()
    X_super, y_super = oversample.fit_resample(x_train,y_train)

    # # print("Class counts for the supersampled dataset")
    # y_supersampled_df = pd.DataFrame(data = y_super, columns=["class"])
    # # print(y_supersampled_df['class'].value_counts())

    # test_predictions, train_predictions = [], []
    # looc_test_predictions, looc_train_predictions = [], []
    # supersampled_predictions = {}

    # distance_functions = ['cosine','braycurtis','sqeuclidean']
    # ll = 2

    # #Comparison of 3 models 1. Normal, 2. Balanced 3. Balanced with PCA
    clf_normal = KnnClassifier(n_neighbors=5)
    clf_supersampled = KnnClassifier(n_neighbors=5)
    clf_supersample_pca = KnnClassifier(n_neighbors=5)

    pca = PCA(n_components=50)
    x_train_pca = pca.fit_transform(X_super)
    x_test_pca = pca.transform(x_test)
    # # loop over number of neighbors
    # start_time = datetime.now() 
    # # for distance in distance_functions:

    clf_normal.fit(x_train, y_train)
    clf_supersampled.fit(X_super,y_super)
    clf_supersample_pca.fit(x_train_pca,y_train)


    # # normal parallel

    # # accuracy_train = clf.accuracy_score(clf.predict_parallel(x_test,distance,2), y_test)
    distance_metrics = ["euclidean_distance","canberra","chebyshev","braycurtis","cosine","seuclidean"]
    for distance in distance_metrics:
        print("Start model 1 ")
        accuracy_test = clf_normal.accuracy_score_old(clf_normal.predict_parallel_old(x_test,distance), y_test)
        print("Start model 2 ")
        accuracy_test_supersample = clf_supersampled.accuracy_score_old(clf_supersampled.predict_parallel_old(x_test,distance),y_test)
        print("Start model 3 ")
        accuracy_test_supersample_pca = clf_supersample_pca.accuracy_score_old(clf_supersample_pca.predict_parallel_old(x_test_pca,distance),y_test)

        print("Normal ", accuracy_test)
        print("Supersample ", accuracy_test_supersample)
        print("Supersample PCA ", accuracy_test_supersample_pca)
        # # supersampled_predictions.append(accuracy_train_supersampled)
    # # train_predictions.append(accuracy_train)
    # # supersampled_predictions.append({"Distance Function" : distance, "Loss": accuracy_train_supersampled})

    # time_elapsed = datetime.now() - start_time 

    # print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    # # plot_question_D(supersampled_predictions)
    

if __name__ == "__main__":
    main()