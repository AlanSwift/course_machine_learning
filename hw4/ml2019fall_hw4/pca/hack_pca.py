import numpy as np
import matplotlib.pyplot as plt
from pca import PCA

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)
    

    # YOUR CODE HERE
    # begin answer
    idx = np.where(img_r > 0)
    idx = np.array(idx)
    w, v = PCA(idx.T)
    v_rev = np.linalg.inv(v)
    vv = np.array([1, 0 , 0])
    t = np.matmul(v_rev, vv)
    angle = np.arctan(t[0] / t[1]) * 180 / np.pi
    from scipy import misc
    return misc.imrotate(img_r, angle)
    # end answer