import numpy as np
import kmeans as kmeans

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    [n_node, d_f] = list(W.shape)
    D = np.diagflat(W.sum(-1))
    L = D - W
    D_rev = np.diagflat(1 / W.sum(-1))
    D_L = np.matmul(D_rev, L)
    w, v = np.linalg.eig(D_L)
    
    index = np.argsort(w)[0:k]
    v_k = v[:, index]
    # v_k = np.dot(np.sqrt(D), v_k)
    v_k = np.array(v_k)
    idx = kmeans.kmeans(v_k, k)
    # end answer
    return idx
