import numpy as np
import random as rd


def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE

    # begin answer
    [n_points, p] = list(x.shape)
    ctrs = x[rd.sample([i for i in range(n_points)], k), :]
    iters = []
    for i in range(1000):
        x_ = x.reshape(n_points, 1, p).repeat(k, axis=1)
        pre = ctrs
        iters.append(pre)
        pre_ctrs = ctrs.reshape(1, k, p).repeat(n_points, axis=0)
        diff = x_ - pre_ctrs
        dis = np.sqrt(np.sum(diff * diff, axis=-1))
        idx = np.argsort(dis, axis=1)[:, 0]
        ctrs = np.zeros((k, p))
        cnt = np.zeros((k))
        for j in range(n_points):
            ctrs[idx[j], :] += x[j, :]
            cnt[idx[j]] += 1
        for j in range(k):
            ctrs[j] /= cnt[j] + 1e-8
        if (ctrs == pre).all():
            break
    iter_ctrs = np.array(iters)
    # end answer

    return idx, ctrs, iter_ctrs
