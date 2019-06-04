import numpy as np
import matplotlib.pyplot as plt
import scipy.io as mat
import torch
import torch.nn.functional as op
from layers import Convolutional, Pooling, Dense
from CNN import CNN
from functional import dataSet, Cost

if __name__ == "__main__":

    
    raw_data = mat.loadmat("mnist_reduced.mat")

    X = raw_data['X']
    X_shape = X.shape
    X = torch.tensor(X).type(torch.FloatTensor)
    X = X.view(X_shape[0],1,20,20)

    X = X.permute(0,1,3,2)
    """
    fig, ax = plt.subplots(nrows=2, ncols=5)
    c=0
    for row in ax:
        for col in row:
            col.imshow(X[c][0])
            c+=500

    plt.show()"""


    y = raw_data['y'].ravel()

    Y = np.zeros((5000, 10), dtype='uint8')

    for i, row in enumerate(Y):
        Y[i, y[i] - 1] = 1

    y = torch.tensor(y).type(torch.LongTensor)
    Y = torch.tensor(Y).type(torch.FloatTensor)

    data = dataSet(X,y,(20,20,1),Y)

    NN_ARCHITECTURE = [
        {"type": "POOL","p_type":"AVG",  "filter_shape": (2,2),"stride":2,"padding":0},
        {"type": "Dense","activation":"sigmoid", "no_logits": 10},
    ]

    q = CNN(NN_ARCHITECTURE,data)
    q.forward()
    print(q.cost()*100)
    q.backward()