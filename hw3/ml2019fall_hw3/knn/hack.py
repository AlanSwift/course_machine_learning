import numpy as np

import knn
import show_image
import extract_image

def hack(img_name):
    '''
    HACK Recognize a CAPTCHA image
      Inputs:
          img_name: filename of image
      Outputs:
          digits: 1x5 matrix, 5 digits in the input CAPTCHA image.
    '''
    # data = np.load('hack_data.npy', allow_pickle=True)
    import pickle
    with open("data_mat.pkl", "rb") as f:
        data = pickle.load(f)

    # YOUR CODE HERE (you can delete the following code as you wish)
    x_train = data['x_train']
    y_train = data['y_train']
    
    x = extract_image.extract_image(img_name)
    ret = knn.knn(x, x_train, y_train, 10)
    digits = ret

    # begin answer
    # end answer

    return digits
