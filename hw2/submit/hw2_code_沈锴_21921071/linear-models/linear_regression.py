import numpy as np

def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    yy = y.copy()
    y = y.copy()
    y[y == -1] = 0
    
    yy[yy == 1] = 0
    yy[yy == -1] = 1
    
    x = np.vstack((np.ones((1, X.shape[1])), X))
    w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x, x.T)), x), y.T)
    
    ww = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x, x.T)), x), (yy).T)
    
    # end answer
    return w - ww
