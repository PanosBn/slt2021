import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
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


def plot_pca(normal_test_predictions,pca_test_predictions, params):
    fig = plt.figure()
    fig.set_figwidth(20)
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(params, normal_test_predictions, label=" Normal Set", marker = '.')
    ax1.plot(params, pca_test_predictions, label="PCA Set ", marker = '+') 
    ax1.set_xticks(params)
    ax1.legend()
    plt.show()




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

    
    # x_test_pca = PCA(n_components=175).fit_transform(x_t)

    normal_test_predictions, pca_test_predictions = [], []
 
    component_range = [25,50,75,100,125,150,175,200,225,250,275,300]    # Do a search just for the first 300 components because 
    # component_range = [25,50,75,100]                                  # Because of the stuff we see in the explained variance plot
    ll = 5
    for num in component_range:
        for k in tqdm(range(4,ll)):
            pca = PCA(n_components=num)
            x_train_pca = pca.fit_transform(x_train)
            clf_normal = KnnClassifier(n_neighbors=k)
            clf_normal.fit(x_train,y_train)
            clf_pca = KnnClassifier(n_neighbors=k)
            clf_pca.fit(x_train_pca, y_train)
            x_test_pca = pca.transform(x_test)
            print("\t Started PCA testing for ", num , " principal components")
            # clf = KnnClassifier(n_neighbors=7)

            # clf.fit(x_train, y_train)


            accuracy_test = clf_normal.accuracy_score(clf_normal.predict_parallel(x_test,2), y_test)
            accuracy_test_pca = clf_pca.accuracy_score(clf_pca.predict_parallel(x_test_pca,2),y_test)
            normal_test_predictions.append(accuracy_test)
            pca_test_predictions.append(accuracy_test_pca)


    plot_pca(normal_test_predictions,pca_test_predictions,component_range)
    # # Write results to a csv for later use
    # table_for_q_a = pd.DataFrame({'Number of components' : train_predictions, 'loss' : test_predictions})
    # table_for_q_a.to_csv('pca_component_search.csv',index=False)



def main():
    # initialize datasets from .csv files:
    train_small = pd.read_csv("data/MNIST_train_small.csv", nrows=3000)
    test_small  = pd.read_csv("data/MNIST_test_small.csv", nrows=1000)

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