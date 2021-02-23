
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from metrics import euclidean_distance, accuracy_score
from KnnClassifier import KnnClassifier
from tqdm import tqdm


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
    # initialize datasets from .csv files:
    train_small = pd.read_csv("data/MNIST_train_small.csv")
    test_small  = pd.read_csv("data/MNIST_test_small.csv")

    train = pd.read_csv("data/MNIST_train.csv")
    test  = pd.read_csv("data/MNIST_test_small.csv")

    
    # split both datasets to digits and labels (the first item in every row is a label):
    # x_train = train_small.values[:,1:]
    # y_train = train_small.values[:,0]
    # x_test = test_small.values[:,1:]
    # y_test = test_small.values[:,0]
    

    # or pick smaller subsets of the dataset:
    x_train = train.values[:2900,1:]
    y_train = train.values[:2900,0]
    x_test = test.values[:900,1:]
    y_test = test.values[:900,0]
    
    '''
    # show a summary
    train.info()
    test.info()
    '''

    # clf = KnnClassifier(x_train,y_train,5).predict(x_test,y_test)
    # k = 5
    y_train_predictions_set = []
    y_test_predictions_set= []
    y_test_prediction_set_errors = []
    y_train_predictions_set_errors= []
    y_train_loss = 0
    y_test_loss = 0
    ll = 11
    for k in tqdm(range(1,ll)):
        clf = KnnClassifier(x_train,y_train)
        # y_train_predictions = clf.predict(x_train[:50])
        y_test_predictions = clf.predict(x_test[:90],k)
        y_train_predictions = clf.predict(x_train[:290],k)
        # print(y_test_predictions)
        accuracy_test = accuracy_score(y_test[:90], y_test_predictions[:90])
        accuracy_train = accuracy_score(y_train[:290], y_train_predictions[:290])
        y_test_predictions_set.append(accuracy_test)
        y_train_predictions_set.append(accuracy_train)

        # y_test_prediction_set_errors.append(accuracy_test_errors)
        # y_train_predictions_set_errors.append(accuracy_test_errors)
        # print(train_accuracy)
        # y_test_predictions = clf.predict(x_train[:100])
    
    for item in y_train_predictions_set:
        print(type(item))
    for item in y_test_predictions_set:
        print(type(item))
    # y_train_predictions_set = y_train_predictions_set.sort()

    # y_test_predictions_set = y_test_predictions_set.sort()
    # df_result = pd.DataFrame()
    # df_result['K value'] = [1,2,3]
    # df_result['train pred'] = y_train_predictions_set
    # df_result['test pred'] = y_test_predictions_set
    # print(df_result)

    plot_q3(y_train_predictions_set, y_test_predictions_set, ll-1)
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