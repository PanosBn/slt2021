import numpy as np

def euclidean_distance(row1, row2): return np.sqrt(np.sum((row1 - row2)**2))

# def accuracy_score(y_test, y_pred): return (y_test == y_pred).mean()

def accuracy_score(a, b) :
    # total_error = sum(1 for i, j in zip(a, b) if i != j)
    # print("a is a:" , type(a))
    # return total_error
    total_error = 0
    for i,j in zip(a,b):
        if i != j: 
            print(i, "<-predicted, true answer ->", j)
            total_error = total_error + 1
        elif (i == j):
            print("\tCorrect Prediction")
            print("\t",i, "<-predicted, true answer ->", j)
    if(total_error == 0):
        return 1
    else:
        return len(a)-total_error/len(a)
