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
    y = torch.tensor(y).type(torch.LongTensor)

    data = dataSet(X,y,(20,20,1))

    NN_ARCHITECTURE = [
        {"type": "CONV", "activation":"relu","filter_shape": (5,5),"no_channels":6,"stride":1,"padding":0},
        {"type": "POOL","p_type":"AVG", "filter_shape": (2,2),"stride":2,"padding":0},
        {"type": "CONV", "activation":"relu","filter_shape": (5,5),"no_channels":12,"stride":1,"padding":0},
        {"type": "POOL","p_type":"AVG",  "filter_shape": (2,2),"stride":2,"padding":0},
        {"type": "Dense","activation":"sigmoid", "no_logits": 10},
    ]

    q = CNN(NN_ARCHITECTURE,data)
    q.forward()
    print(q.cost()*100)