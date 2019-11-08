import numpy as np

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    def gaosi(x, mu, sigma):
        '''
            x: [M]
            mu:[M]
            Sigma:[M, M]
        '''
        sig = np.matrix(sigma)
        zs = np.matmul((x - mu).T, sig.I)
        zs = np.matmul(zs, (x - mu))
        fenmu = np.sqrt(np.linalg.det(sig)) * 2 * np.pi
        return np.exp(-0.5*zs) / fenmu
    
    for i in range(N):
        
        for j in range(K):
            p[i, j] = Phi[j] * gaosi(X[:, i], Mu[:, j], Sigma[:, :, j])
        s = np.sum(p[i, :])
        p[i, :] /= s
    
    # end answer
    
    return p
    