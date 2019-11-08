import numpy as np

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    [n_node, d_f] = list(X.shape)
    mask = np.zeros((n_node, n_node))
    for i in range(n_node):
        t = X[i, :]
        t = t.reshape(1, d_f).repeat(n_node, 0)
        dis = np.sum((t - X) ** 2, -1)
        sorted_index = dis.argsort()
        for j in range(k+1):
            index = sorted_index[j]
            if i == index:
                continue
            if dis[index] < threshold ** 2:
                break
            mask[i, index] = 1
            mask[index, i] = 1
            
    
    return mask
    # end answer
