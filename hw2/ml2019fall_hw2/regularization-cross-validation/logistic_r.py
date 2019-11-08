import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def calc_grad(w, x, y, lmbda=None):
    #print(w.shape, x.shape, y.shape) # 3,1   3, 100   1, 100
    yy = sigmoid(np.matmul(w.T, x))
    yy -= y
    P = w.shape[0]
    yy = np.repeat(yy, P, axis=0)
    grad = yy * x
    grad = np.mean(grad, axis=1)
    return grad + np.mean(w)*lmbda

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    x = np.vstack((np.ones((1, X.shape[1])), X))
    T = 1000
    lr = 0.01
    
    for i in range(T):
        grad = calc_grad(w, x, y, lmbda).reshape((P+1), 1)
        sm = np.sum(grad)
        if sm > 0 and sm < 1e-3:
            break
        #print(grad)
        w -= grad * lr
    # end answer
    return w
