import numpy as np
from scipy.optimize import minimize

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0
    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    x = np.vstack((np.ones((1, X.shape[1])), X))
    
    
    def loss(w):
        return 0.5 * np.sum(w * w)
    
    
    def constraint(i):
        def g(w):
            return y[0, i] * (np.matmul(w.T, x[:, i])) - 1
        return g
    
    cons = ()
    
    for i in range(N):
        cons += ({'type': 'ineq', 'fun': constraint(i)},)
        
    ans = minimize(loss, w, method='SLSQP', constraints=cons)
    w = ans.x
    
    
    dis = np.abs(y * np.matmul(w.T, x))
    eps = 1e-8
    msk = dis <= 1 + eps
    cnt = np.sum(msk)
    
    # end answer
    return w, cnt


