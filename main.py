
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from metrics import euclidean_distance, accuracy_score
from KnnClassifier import KnnClassifier
from tqdm import tqdm


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
    x_train = train.values[:3000,1:]
    y_train = train.values[:3000,0]
    x_test = test.values[:150,1:]
    y_test = test.values[:150,0]
    
    '''
    # show a summary
    train.info()
    test.info()
    '''

    # clf = KnnClassifier(x_train,y_train,5).predict(x_test,y_test)
    # k = 5
    y_train_predictions = []
    y_test_predictions = []
    y_train_loss = 0
    y_test_loss = 0
    for k in tqdm(range(1,21)):
        clf = KnnClassifier(x_train,y_train)
        # y_train_predictions = clf.predict(x_train[:50])
        y_test_predictions = clf.predict(x_test[:50],k)
        print(y_test_predictions)
        train_accuracy = accuracy_score(y_test[:50], y_test_predictions[:50])
        # y_test_predictions = clf.predict(x_train[:100])
      
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