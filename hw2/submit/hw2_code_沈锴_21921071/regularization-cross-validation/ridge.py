import numpy as np

def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
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
    p = np.matmul(np.linalg.pinv(np.matmul(x, x.T) + lmbda * np.eye(P + 1)), x)
    w = np.matmul(p, y.T)
    
    ww = np.matmul(p, (yy).T)
    
    # end answer
    return w - ww
