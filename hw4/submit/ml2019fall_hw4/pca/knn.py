import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer
    # end answer
    [n_test, d] = list(x.shape)
    [n_train, d] = list(x_train.shape)
    y = []
    for i in range(n_test):
        t = x[i, :]
        t = t.reshape(1, -1).repeat(n_train, 0)
        dis = np.sum((t - x_train) ** 2, -1)
        index = dis.argsort()[0:k]
        labels = [y_train[_] for _ in index]
        c = np.argmax(np.bincount(labels))
        y.append(c)
    

    return y
