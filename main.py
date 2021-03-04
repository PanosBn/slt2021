import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime 
from collections import defaultdict
from metrics import *
from KnnClassifier import KnnClassifier
from tqdm import tqdm
import sys
import time
from functools import partial
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

def q_a(X, y, Xt, yt):
    clf = KnnClassifier(distance_function="euclidean_distance")
    clf.fit(X, y)
    acc_train, acc_test = [], []
    k_max = 20
    labels_train = clf.predict_parallel(X, return_multiple=True)
    labels_test = clf.predict_parallel(Xt, return_multiple=True)
    
    for k in tqdm(range(1, k_max + 1)):
        acc_train.append(clf.score(labels_train, y, n_neighbors=k, multiple=True, how="accuracy") * 100)
        acc_test.append(clf.score(labels_test, yt, n_neighbors=k, multiple=True,  how="accuracy") * 100) 
    
    font = {'family': 'Verdana', 'color': 'black', 'weight': 'normal', 'size': 10,}

    fg, (ax1,ax2) = plt.subplots(2, 1)
    ax1.plot(range(1, k_max + 1), acc_train, label="train set")
    ax1.plot(range(1, k_max + 1), acc_test, label="test set")
    ax1.legend()
    ax1.set_title("Accuracy kNN compared to k neighbors", fontdict=font)
    ax1.set_xticks(range(1, k_max + 1))
    ax1.set_xlabel("k neighbors")
    ax1.set_ylabel("Accuracy score in %")

    table = [acc_train, acc_test]

    ax2 = sns.heatmap(table, cmap='BuPu', square=True, annot=True, annot_kws={"size": 6}, fmt='.2f', yticklabels=["train", "test"], xticklabels=range(1, k_max+1), cbar=False)

    plt.show()

def q_b(X, y):
    clf = KnnClassifier(distance_function="euclidean_distance")
    clf.fit(X, y)
    acc_looc, k_neigh = [], []
    k_max = 20
    labels_train = clf.looc_parallel(X, y, return_multiple=True)

    for k in tqdm(range(1, k_max + 1)):
        acc_looc.append(clf.score(labels_train, y, n_neighbors=k, multiple=True, how="loss"))
        k_neigh.append(k) 
    
    font = {'family': 'Verdana', 'color': 'black', 'weight': 'normal', 'size': 10,}

    plt.title("LOOCV accuracy", fontdict=font)
    plt.plot(k_neigh, acc_looc)
    plt.xlabel('k neighbors')
    plt.ylabel('LOOCV accuracy score')
    plt.show()

def q_c():
    pass

def q_d(x_train, y_train, x_test, y_test):

    def plot_question_d(df_results,params):
        g = sns.catplot(
            data=df_results, kind="bar",
            x="Metric", y="test loss", hue="Model",
            ci="sd", palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        g.set_axis_labels("", "Empirical test loss (%)")
        plt.show()

    oversample = SMOTE()
    X_super, y_super = oversample.fit_resample(x_train,y_train)

    clf_normal = KnnClassifier(n_neighbors=5)
    clf_pca = KnnClassifier(n_neighbors=5)
    clf_supersampled = KnnClassifier(n_neighbors=5)
    clf_supersample_pca = KnnClassifier(n_neighbors=5)

    pca = PCA(n_components=50)
    x_train_pca = pca.fit_transform(X_super)
    x_test_pca = pca.transform(x_test)

    clf_normal.fit(x_train, y_train)
    clf_pca.fit(x_train_pca,y_train)
    clf_supersampled.fit(X_super,y_super)
    clf_supersample_pca.fit(x_train_pca,y_train)

    times = {}
    accuracies = {}

    df_results = pd.DataFrame(columns=['Model','Metric','test loss','time'])

    models = [clf_normal, clf_pca, clf_supersampled, clf_supersample_pca]
    model_names = ["Vanilla", "PCA", "Balanced", "Balanced_PCA"]
    distance_metrics = ["manhattan","euclidean_distance","canberra","chebyshev","braycurtis","cosine"]

    for model, name in zip(models, model_names):
        for distance in distance_metrics:
            model.set_distance_func(distance)
            start_time = time.time()
            if ((name == "PCA") or (name == "Balanced_PCA")):
                pred = model.predict_parallel(x_test_pca)
                accuracy_test = model.score(pred, y_test)
            else:
                pred = model.predict_parallel(x_test)
                accuracy_test = model.score(pred, y_test)
            time_elapsed = (time.time() - start_time)

            df_results = df_results.append({'Model':name, 'Metric':distance, 'test loss': accuracy_test, 'time': time_elapsed},ignore_index=True)

    df_results.sort_values('test loss')
    plot_question_d(df_results,distance_metrics)

def q_e():
    pass

def q_f():
    pass

def q_g(x_train, y_train ,x_test,y_test):

    """
    :To do the dimensionality reduction we need to see which of the 784 features actually
    :contribute the most to the model. So we start by plotting the explained variance of all the components(i.e. 784)
    :In the plot we will see that ~175 components contribute around 97% of the explained variance, so we will do a 
    :dimensionality reduction and we will keep only these components, thus reducing each image from 784 datapoints to just 
    :175.
    :An alternative(and probably the best) way to calculate the optimal amount of components would be to do 
    :a search with varying amount of components (i.e. from 1 to 784) and compare the accuracy.
    """

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
        accuracy_test = clf_normal.score(clf_normal.predict_parallel(x_test), y_test, how="accuracy")
        time_elapsed_normal = (time.time() - start_time)
                
        start_time = time.time()
        accuracy_test_pca = clf_pca.score(clf_pca.predict_parallel(x_test_pca), y_test, how="accuracy")
        time_elapsed_pca = (time.time() - start_time) 

        normal_test_predictions.append(accuracy_test)
        normal_test_times.append(time_elapsed_normal)

        pca_test_predictions.append(accuracy_test_pca)
        pca_test_times.append(time_elapsed_pca)



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

def compare_over_kp(x_train, y_train, x_test, y_test, max_k, max_p):
    kp_test_predictions = np.zeros((max_k, max_p))
    kp_train_predictions = np.zeros((max_k, max_p))

    for k in tqdm(range(1,max_k+1)):
        for p in range(1, max_p+1):
            clf = KnnClassifier(n_neighbors=k)
            clf.fit(x_train, y_train)

            kp_accuracy_train = clf.looc_validate_parallel(x_train, y_train, minkowski_distance, p)
            kp_accuracy_test = clf.looc_validate_parallel(x_test, y_test, minkowski_distance, p)
            kp_train_predictions[k-1, p-1] = kp_accuracy_train
            kp_test_predictions[k-1, p-1] = kp_accuracy_test

    print(kp_train_predictions, "\n\n", kp_test_predictions)
    plot_kp(kp_train_predictions, kp_test_predictions, max_k)

def test(X, y):
    p_max = 5
    k_max = 20
    pk = []
    for p in range(1, p_max + 1):
        minowski = partial(minkowski_distance, p=p)

        clf = KnnClassifier(distance_function=minowski)
        labels = clf.looc_parallel(X, y, return_multiple=True, tree_search=False) # use tree search to search with kdtree, this makes it faster, but also increases error
        loss = []
        
        for k in range(1, k_max + 1):
            loss.append(clf.loss_score(labels, y, n_neighbors=k)) 
        pk.append(loss)

    pk = np.array(np.reshape(pk, (len(pk), len(pk[0]))))
    return pk     


def main():
    sys.setrecursionlimit(10000)

    train_small = pd.read_csv("data/MNIST_train_small.csv", nrows=300, header=None)
    test_small  = pd.read_csv("data/MNIST_test_small.csv", nrows=100, header=None)
    
    # split both datasets to digits and labels (the first item in every row is a label):
    x_train = train_small.values[:,1:]
    y_train = train_small.values[:,0]
    x_test = test_small.values[:,1:]
    y_test = test_small.values[:,0]
    

    """
    AFTER THIS YOU CAN CALL YOUR METHODS FOR SEPARATE ASSIGNMENT SUBQUESTIONS
    """
    #q_a(x_train, y_train, x_test, y_test)
    #q_b(x_train, y_train)
    q_d(x_train, y_train, x_test, y_test)
    #q_g(x_train,y_train,x_test,y_test) 


if __name__ == "__main__":
    main()


















def plot_accuracy(y_train_basic, y_test_basic, y_train_loocv, y_test_loocv, max_k):
    """
    IS THIS STILL NEEDED?
    """
    x_range = np.linspace(1, max_k, num=max_k)
    fig = plt.figure()
    fig.set_figwidth(15)
    
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x_range, y_train_basic, label='Training set, basic', marker='.')
    ax1.plot(x_range, y_test_basic, label='Test set, basic', marker='.')
    ax1.plot(x_range, y_train_loocv, label='Training set, LOOC', marker='.')
    ax1.plot(x_range, y_test_loocv, label='Test set, LOOC', marker='.')
    ax1.set_xlabel("Number of neighbours")
    ax1.set_ylabel("Accuracy (%))")
    ax1.set_title("Accuracy basic vs. LOOC")
    ax1.legend()

    plt.show()
    print("test")

def plot_kp(train, test, max_k):
    """
    IS THIS STILL NEEDED?
    """
    x_range = np.linspace(1, max_k, num=max_k)
    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(20)
    _, ncol = train.shape
    axs = {}

    for i in range(0, ncol):
        axs[i] = fig.add_subplot(5, 3, i+1)
        
        axs[i].plot(x_range, train[:,i], label=f'Training set p = {i+1}', marker='.')
        axs[i].plot(x_range, test[:,i], label=f'Test set p = {i+1}', marker='.')

        axs[i].set_xlabel("Number of neighbours")
        axs[i].set_ylabel("Accuracy (%))")
        axs[i].set_title("Accuracy using k, p")
        axs[i].legend()
        
    plt.show()

def compare_over_kp(x_train, y_train, x_test, y_test, max_k, max_p):
    """
    IS THIS STILL NEEDED?
    """
    kp_test_predictions = np.zeros((max_k, max_p))
    kp_train_predictions = np.zeros((max_k, max_p))

    for k in tqdm(range(1,max_k+1)):
        for p in range(1, max_p+1):
            clf = KnnClassifier(n_neighbors=k)
            clf.fit(x_train, y_train)

            kp_accuracy_train = clf.looc_validate_parallel(x_train, y_train, minkowski_distance, p)
            kp_accuracy_test = clf.looc_validate_parallel(x_test, y_test, minkowski_distance, p)
            kp_train_predictions[k-1, p-1] = kp_accuracy_train
            kp_test_predictions[k-1, p-1] = kp_accuracy_test

    print(kp_train_predictions, "\n\n", kp_test_predictions)
    plot_kp(kp_train_predictions, kp_test_predictions, max_k)




def test(X, y):
    """
    IS THIS STILL NEEDED?
    """
    p_max = 15
    k_max = 20
    pk = []
    for p in range(1, p_max + 1):
        minowski = partial(minkowski_distance, p=p)

        clf = KnnClassifier(distance_function=minowski)
        labels = clf.looc_parallel(X, y, return_multiple=True, tree_search=False) # use tree search to search with kdtree, this makes it faster, but also increases error
        loss = []
        
        for k in range(1, k_max + 1):
            loss.append(clf.loss_score(labels, y, n_neighbors=k)) 
        pk.append(loss)

    pk = np.array(np.reshape(pk, (len(pk), len(pk[0]))))
    return pk   

def heat_pk(pk):
    """
    IS THIS STILL NEEDED?
    """
    
    font = {'family': 'Verdana', 'color': 'black', 'weight': 'normal', 'size': 10,}
    ax = sns.heatmap(pk, cmap='BuPu', square=True, annot=True, annot_kws={"size": 6}, fmt='.2f')
    plt.title("Loss for (k, p)", fontdict=font)
    plt.xlabel('k', fontdict=font)
    plt.ylabel('p', fontdict=font)
    ax.invert_yaxis()
    #plt.axis([1, 20, 1, 15])
    plt.show()