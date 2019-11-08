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
    [n_test, p] = list(x.shape)
    [n, p] = list(x_train.shape)
    x_ = x.reshape(n_test, 1, p).repeat(n, 1)
    x_train_ = x_train.reshape(1, n, p).repeat(n_test, 0)
    diff = x_ - x_train_
    dis = np.sum(diff * diff, axis=-1)
    idx = np.argsort(dis, axis=1)
    idx = idx[:, 0:k]
    max_label = np.max(y_train) + 1
    y = np.zeros((n_test))
    for i in range(n_test):
        votes = np.zeros((max_label))
        for j in range(k):
            votes[y_train[idx[i, j]]] += 1
        vote = np.argmax(votes)
        y[i] = vote
    
    # end answer

    return y
