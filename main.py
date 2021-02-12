import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

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

    #def euclidian_distance(vector1, vector2):

    #def fit():

    #def predict(x_train, y_train, x_test, y_test, find_neihgbours()):

    #def knn_classifier(x_train, y_train, x_test, y_test, number_of_neighbours)


if __name__ == "__main__":
    main()