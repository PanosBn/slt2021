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


def plot_q3(normal_test_predictions, pca_test_predictions, l):
    x_range = np.linspace(1, l, num=l)
    fig = plt.figure()
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(x_range, normal_test_predictions, label='Normal set', marker='.')
    ax1.plot(x_range, pca_test_predictions, label='PCA set', marker='.')
    plt.ylim([70,100])
    ax1.set_xlabel("Number of neighbours")
    ax1.set_ylabel("Loss")
    ax1.set_title("Comparison between normalloss")
    ax1.legend()
    
    # ax2.plot(x_range, grd_performance, label='Gradient Descent')
    # ax2.plot(x_range, quad_performance, label='Quadratic')
    # ax2.set_xlabel("Dataset size (%)")
    # ax2.set_ylabel("Testing accuracy (%)")
    # ax2.set_title("Testing Accuracy")
    # ax2.legend()
    plt.show()


# def plot_pca(pca_test_predictions, normal_test_predictions,params):






def pca_q(x_train, y_train ,x_test,y_test):

    """
    :To do the dimensionality reduction we need to see which of the 784 features actually
    :contribute the most to the model. So we start by plotting the explained variance of all the components(i.e. 784)
    :In the plot we will see that ~175 components contribute around 97% of the explained variance, so we will do a 
    :dimensionality reduction and we will keep only these components, thus reducing each image from 784 datapoints to just 
    :175.
    :An alternative(and probably the best) way to calculate the optimal amount of components would be to do 
    :a search with varying amount of components (i.e. from 1 to 784) and compare the accuracy.
    """
    # pca = PCA().fit(x_train)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('number of components')
    # plt.ylabel('cumulative explained variance');
    # plt.show()

    
    # x_train_pca = PCA(n_components=25).fit_transform(x_train)

    normal_test_predictions, pca_test_predictions, normal_test_times, pca_test_times = [], [], [], []
 
    component_range = [25,50,75,100,125,150,175,200,225,250,275,300]    # Do a search just for the first 300 components 
    # component_range = [25,50,75,100]                                 # Because of the stuff we see in the explained variance plot

    for num in tqdm(component_range):
        pca = PCA(n_components=num)
        x_train_pca = pca.fit_transform(x_train)
        clf_normal = KnnClassifier(n_neighbors=4)
        clf_normal.fit(x_train,y_train)
        clf_pca = KnnClassifier(n_neighbors=4)
        clf_pca.fit(x_train_pca, y_train)

        x_test_pca = pca.transform(x_test)

        start_time = time.time()
        accuracy_test = clf_normal.accuracy_score_old(clf_normal.predict_parallel_old(x_test,2), y_test)
        time_elapsed_normal = (time.time() - start_time)
        print("\tNormal Set, time elapsed: ", time_elapsed_normal)
                
        start_time = time.time()
        accuracy_test_pca = clf_pca.accuracy_score_old(clf_pca.predict_parallel_old(x_test_pca,2),y_test)
        time_elapsed_pca = (time.time() - start_time) 
        print("\tPCA Set for ", num, " components. Time elapsed: ", time_elapsed_pca)

        normal_test_predictions.append(accuracy_test)
        normal_test_times.append(time_elapsed_normal)

        pca_test_predictions.append(accuracy_test_pca)
        pca_test_times.append(time_elapsed_pca)


    # plot_pca(pca_test_predictions,normal_test_predictions,component_range)


    fig = plt.figure()
    fig.set_figwidth(20)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.plot(component_range, normal_test_predictions, label=" Normal Set - All components", marker = '.')
    ax1.plot(component_range, pca_test_predictions, label="PCA Set ", marker = '+') 
    ax1.set_xticks(component_range)
    bottom = min(pca_test_predictions)
    top = max(pca_test_predictions)
    ax1.set_ylim(bottom-0.05, top+0.05)
    ax1.legend()

    # ax2.    ax1.plot(component_range, normal_test_predictions, label=" Normal Set - All components", marker = '.')
    ax2.plot(component_range, normal_test_times, label="Normal Set", marker ='.')
    ax2.plot(component_range, pca_test_times, label="PCA Set ", marker = '+') 
    ax2.set_xticks(component_range)
    n = max(normal_test_times)
    p = max(pca_test_times)
    bottom = min(n,p)
    top = max(n,p)
    ax2.set_ylim(0, top)
    ax2.legend()

    plt.show()




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


    pca_q(x_train,y_train,x_test,y_test) #      <-----  Subquestion g


if __name__ == "__main__":
    main()