import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    
    # begin answer
    # print(X.shape, w.shape, y.shape)
    # (2, 10) (3, 1) (1, 10)
    while True:
        iters += 1
        x = np.vstack((np.ones((1, N)), X))
        ok = True
        for i in range(N):
            yy = np.sign(np.matmul(w.T, x[:, i]))
            if yy[0] != y[0, i]:
                tmp = x[:, i] * y[0, i]
                w += tmp.reshape((P+1), 1)
                ok = False
        if ok:
            break
        

    # end answer
    
    return w, iters