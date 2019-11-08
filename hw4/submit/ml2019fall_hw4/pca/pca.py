import numpy as np

def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer
    [n_node, d_f] = list(data.shape)
    data_ = data - np.mean(data, axis = 0)
    s = np.matmul(data_.T, data_) / n_node
    w, v = np.linalg.eig(s)
    index = np.argsort(-w)
    v_ = v[:, index]
    w_ = w[index]
    return w_, v_
    
    # end answer