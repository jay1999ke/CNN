import numpy as np
import matplotlib.pyplot as plt
import scipy.io as mat
import torch
import torch.nn.functional as op
from layers import Convolutional, Pooling, Dense
from CNN import CNN
from functional import dataSet, Cost
from sklearn.utils import shuffle

if __name__ == "__main__":

    
    raw_data = mat.loadmat("mnist_reduced.mat")

    X = raw_data['X']
    y = raw_data['y'].ravel()
    X, y = shuffle(X, y, random_state=0)

    X_train = X[:4000]
    X_test = X[4000:]
    X = X_train

    X_shape = X.shape
    X_test_shape = X_test.shape
    print(X_test_shape)

    X = torch.tensor(X).type(torch.FloatTensor)
    X_test = torch.tensor(X_test).type(torch.FloatTensor)
    
    X = X.view(X_shape[0],1,20,20)
    X_test = X_test.view(X_test_shape[0],1,20,20)

    X = X.permute(0,1,3,2)
    X_test = X_test.permute(0,1,3,2)
    """
    fig, ax = plt.subplots(nrows=2, ncols=5)
    c=0
    for row in ax:
        for col in row:
            col.imshow(X[c][0])
            c+=500

    plt.show()"""


    y_train = y[:4000]
    y_test = y[4000:]

    Y = np.zeros((5000, 10), dtype='uint8')

    for i, row in enumerate(Y):
        Y[i, y[i] -1] = 1

    y = y_train
    Y_train = Y[:4000]
    Y_test = Y[4000:]
    Y = Y_train

    y = torch.tensor(y).type(torch.LongTensor)
    Y = torch.tensor(Y).type(torch.FloatTensor)
    y_test = torch.tensor(y_test).type(torch.LongTensor)
    Y_test = torch.tensor(Y_test).type(torch.FloatTensor)
    data = dataSet(X,y,(20,20,1),Y,X_test,y_test,Y_test)

    NN_ARCHITECTURE = [
        {"type": "CONV", "activation":"relu","filter_shape": (5,5),"no_channels":6,"stride":1,"padding":0},
        {"type": "CONV", "activation":"relu","filter_shape": (3,3),"no_channels":12,"stride":1,"padding":0},
        {"type": "Dense","activation":"softmax", "no_logits": 10},
    ]

    """
    NN_ARCHITECTURE = [
        {"type": "CONV", "activation":"relu","filter_shape": (5,5),"no_channels":6,"stride":1,"padding":0},
        {"type": "POOL","p_type":"MAX", "filter_shape": (2,2),"stride":2,"padding":0},
        {"type": "CONV", "activation":"relu","filter_shape": (3,3),"no_channels":12,"stride":1,"padding":0},
        {"type": "POOL","p_type":"MAX",  "filter_shape": (2,2),"stride":1,"padding":0},
        {"type": "Dense","activation":"sigmoid", "no_logits": 10},
    ]
    """

    q = CNN(NN_ARCHITECTURE,data)
    q.fit(0.00001,2000)

