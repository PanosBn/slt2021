import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime 
from collections import defaultdict
from metrics import euclidean_distance, minkowski_distance
from KnnClassifier import KnnClassifier
from tqdm import tqdm
from sklearn.decomposition import PCA


def plot_q3(y_train_predictions, y_test_predictions, l):
    x_range = np.linspace(1, l, num=l)
    fig = plt.figure()
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(x_range, y_train_predictions, label='Normal set', marker='.')
    ax1.plot(x_range, y_test_predictions, label='PCA set', marker='.')
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



def pca_q(X, y):

    """
    :To do the dimensionality reduction we need to see which of the 784 features actually
    :contribute the most to the model. So we start by plotting the explained variance of all the components(i.e. 784)
    :In the plot we will see that ~175 components contribute around 97% of the explained variance, so we will do a 
    :dimensionality reduction and we will keep only these components, thus reducing each image from 784 datapoints to just 
    :175.
    :An alternative(and probably the best) way to calculate the optimal amount of components would be to do 
    :a search with varying amount of components (i.e. from 1 to 784) and compare the accuracy.
    """
    # pca = PCA().fit(X)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('number of components')
    # plt.ylabel('cumulative explained variance');
    # plt.show()

    x_train = X 
    y_train = y
    x_train_pca = PCA(n_components=175).fit_transform(X)
    # x_test_pca = PCA(n_components=175).fit_transform(x_t)


    train_predictions, train_predictions_pca, test_predictions_pca = [], [], []

    start_time = datetime.now() 
    ll = 11
    for k in tqdm(range(1,ll)):
        print("\t Started PCA testing for ", k , " neighbours")
        clf = KnnClassifier(n_neighbors=k)
        clf_pca = KnnClassifier(n_neighbors=k)
        clf.fit(x_train, y_train)
        clf_pca.fit(x_train_pca, y_train)

        accuracy_train_normal = clf.accuracy_score(clf.predict_parallel(x_train), y_train)
        accuracy_train_pca= clf_pca.accuracy_score(clf_pca.predict_parallel(x_train_pca), y_train)
        train_predictions.append(accuracy_train_normal)
        train_predictions_pca.append(accuracy_train_pca)

    plot_q3(train_predictions, train_predictions_pca, ll-1)


def main():
    # initialize datasets from .csv files:
    train_small = pd.read_csv("data/MNIST_train_small.csv", nrows=1000)
    test_small  = pd.read_csv("data/MNIST_test_small.csv", nrows=500)

    # split both datasets to digits and labels (the first item in every row is a label):
    x_train = train_small.values[:,1:]
    y_train = train_small.values[:,0]
    x_test = test_small.values[:,1:]
    y_test = test_small.values[:,0]

    print("X_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)
 

    # y_train_df = pd.DataFrame(data = y_train, columns=["class"])
    # print(y_train_df['class'].value_counts())

    pca_q(x_train,y_train)

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

    ll = 5
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

        # print("Train predictions: ", type(train_predictions))
        # print("Test predictictions: ", type(test_predictions))

        # looc
        # looc_accuracy_train = clf.looc_validate(x_train, y_train)
        # looc_accuracy_test = clf.looc_validate(x_test, y_test)
        # looc_test_predictions.append(looc_accuracy_test)
        # looc_train_predictions.append(looc_accuracy_train)

        # looc parallel
        # looc_accuracy_train = clf.looc_validate_parallel(x_train, y_train)
        # looc_accuracy_test = clf.looc_validate_parallel(x_test, y_test)
        # looc_test_predictions.append(looc_accuracy_test)
        # looc_train_predictions.append(looc_accuracy_train)

        # Normal with PCA


    time_elapsed = datetime.now() - start_time 

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    


    # plot graphs train vs test score vs n neighbors
    # plot_q3(train_predictions, test_predictions, ll-1)
    # plot_q3(looc_train_predictions, looc_test_predictions, ll-1)


    


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