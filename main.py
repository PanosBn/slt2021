import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    # df = pd.read_csv("data/MNIST_train_small.csv", index_col=False)
    # smaller_df = df[:5]
    # print (df.shape)
    # print (df.describe)
    # print (df.dtypes)

    # print(smaller_df)

    train = pd.read_csv("data/MNIST_train_small.csv")
    test  = pd.read_csv("data/MNIST_test_small.csv")

    #pick a smaller subset of the dataset
    x_train = train.values[:100,1:]
    y_train = train.values[:100,0]

    print(x_train)
    print(y_train)

    # plt.imshow(x_train[4].reshape((28, 28)))
    # plt.show()

    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    #plot 5 random sets of 0 - 9 digits in 5 subplots and add in one big plot
    for y, cls in enumerate(classes):
        idxs = np.nonzero([i == y for i in y_train])
        idxs = np.random.choice(idxs[0], 5)
        for i , idx in enumerate(idxs):
            plt_idx = i * len(classes) + y + 1
            plt.subplot(5, len(classes), plt_idx)
            plt.imshow(x_train[idx].reshape((28, 28)))
            plt.axis("off")
            if i == 0:
                plt.title(cls)
            

    plt.show()

    # ADD SOME BASIC KNN FUNCTIONALITY 

    #def find_neighbours(x_train, number_of_neighbours) -> int:

    # distance between two vectors we mean two rows
    def euclidian_distance(row1, row2): return np.sqrt(np.sum(row1 - row2)**2)

    # find most common labels of the k neighbours
    def majority_of_knn(y_train):
        counter = defaultdict(int)
        for label in y_train:
            counter[label]+=1
        majority_count = max(counter.values())
        for key, value in counter.items():
            if value == majority_count:
                return key

    #def fit():

    def predict(k, x_train, y_train, x_test, y_test):
        distances = [(euclidian_distance(x_test, x_train), label) for (row, label) in zip(x_train, y_train)]
        sorted_distances = sorted(distances, key=lambda (distance, _): distance)

        k_labels = [label for (_, label) in sorted_distances[:k]]
        return majority_of_knn(k_labels)

    #def knn_classifier(x_train, y_train, x_test, y_test, number_of_neighbours):
    
    i = 0
    total_correct = 0
    for test_row in x_test:
        pred = predict(10, x_train, y_train, test_row)
        if pred == y_test[i]:
            total_correct += 1
        acc = (total_correct / (i+1)) * 100
        print('test image['+str(i)+']', '\tpred:', pred, '\torig:', test_labels[i], '\tacc:', str(round(acc, 2))+'%')
        i += 1

if __name__ == "__main__":
    main()