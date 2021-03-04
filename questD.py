import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import time
from datetime import datetime 
from collections import defaultdict
from KnnClassifier import KnnClassifier
import sys
from tqdm import tqdm
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


def plot_question_d(df_results,params):
    g = sns.catplot(
        data=df_results, kind="bar",
        x="Metric", y="test loss", hue="Model",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Empirical test loss (%)")
    plt.show()

def main():

    sys.setrecursionlimit(10000)
    # initialize datasets from .csv files:
    train_small = pd.read_csv("data/MNIST_train_small.csv", nrows=3000, header=None)
    test_small  = pd.read_csv("data/MNIST_test_small.csv", nrows=1000, header=None)

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

    clf_normal = KnnClassifier(n_neighbors=5)
    clf_pca = KnnClassifier(n_neighbors=5)
    clf_supersampled = KnnClassifier(n_neighbors=5)
    clf_supersample_pca = KnnClassifier(n_neighbors=5)

    pca = PCA(n_components=50)
    x_train_pca = pca.fit_transform(X_super)
    x_test_pca = pca.transform(x_test)
    # # loop over number of neighbors
    # start_time = datetime.now() 
    # # for distance in distance_functions:

    clf_normal.fit(x_train, y_train)
    clf_pca.fit(x_train_pca,y_train)
    clf_supersampled.fit(X_super,y_super)
    clf_supersample_pca.fit(x_train_pca,y_train)

    times = {}
    accuracies = {}

    df_results = pd.DataFrame(columns=['Model','Metric','test loss','time'])

    models = [clf_normal, clf_pca, clf_supersampled, clf_supersample_pca]
    model_names = ["Vanilla", "PCA", "Balanced", "Balanced_PCA"]
    # model_names = ["Normal","Balanced_PCA" ]
    distance_metrics = ["manhattan","euclidean_distance","canberra","chebyshev","braycurtis","cosine"]
    # distance_metrics = ["canberra","euclidean_distance"]

    for model, name in zip(models, model_names):
        for distance in distance_metrics:
            start_time = time.time()
            if ((name == "PCA") or (name == "Balanced_PCA")):
                accuracy_test = model.accuracy_score_old(model.predict_parallel_old(x_test_pca,distance), y_test)
            else:
                accuracy_test = model.accuracy_score_old(model.predict_parallel_old(x_test,distance), y_test)
            time_elapsed = (time.time() - start_time)

            df_results = df_results.append({'Model':name, 'Metric':distance, 'test loss': accuracy_test, 'time': time_elapsed},ignore_index=True)

    df_results.sort_values('test loss')
    plot_question_d(df_results,distance_metrics)
    print(df_results)

if __name__ == "__main__":
    main()