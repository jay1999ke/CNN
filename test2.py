"""Test for CIFAR-10 dataset"""

import numpy as np
import torch
import torch.nn.functional as op
from CNN import CNN

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == "__main__":

    
    file = "data_batch_1"

    raw_data = unpickle(file)

    print(raw_data.keys())

    X = raw_data[b"data"]/255
    y = np.array(raw_data[b"labels"])

    Y = np.zeros((10000, 10), dtype='uint8')

    for i, row in enumerate(Y):
        Y[i, y[i] -1] = 1

    X = torch.tensor(X).type(torch.FloatTensor)
    X = X.view(X.size()[0],3,32,32)
    y = torch.tensor(y).type(torch.LongTensor)
    Y = torch.tensor(Y).type(torch.FloatTensor)

    X_train = X[:4500]
    X_test = X[4500:5000]
    X = X_train

    y_train = y[:4500]
    y_test = y[4500:5000]
    y = y_train

    Y_train = Y[:4500]
    Y_test = Y[4500:5000]
    Y = Y_train

    data = dataSet(X,y,(32,32,3),Y,X_test,y_test,Y_test)

    NN_ARCHITECTURE = [
        {"type": "CONV", "activation":"relu","filter_shape": (9,9),"no_channels":3,"stride":1,"padding":0},
        {"type": "POOL","p_type":"MAX", "filter_shape": (2,2),"stride":2,"padding":0},
        {"type": "CONV", "activation":"relu","filter_shape": (7,7),"no_channels":6,"stride":1,"padding":0},
        {"type": "POOL","p_type":"MAX",  "filter_shape": (2,2),"stride":2,"padding":0},
        {"type": "Dense","activation":"softmax", "no_logits": 10},
    ]

    q = CNN(NN_ARCHITECTURE,data)
    q.fit(0.00014,2000)






