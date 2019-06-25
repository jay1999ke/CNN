import numpy as np
import matplotlib.pyplot as plt
import scipy.io as mat
import torch
import torch.nn.functional as op
from layers import Convolutional, Pooling, Dense
from CNN import CNN
from functional import dataSet, Cost
from sklearn.utils import shuffle
import tensorflow as tf

if __name__ == "__main__":

    mnist = tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test) = mnist.load_data()

    X_train_shape = x_train.shape
    x_test_shape = x_test.shape
    print(x_test_shape)

    X = torch.tensor(x_train).type(torch.FloatTensor)
    X_test = torch.tensor(x_test).type(torch.FloatTensor)
    
    X = X.view(x_train_shape[0],1,28,28)
    X_test = X_test.view(X_test_shape[0],1,28,28)

    X = X.permute(0,1,3,2)
    X_test = X_test.permute(0,1,3,2)

    Y_train = np.zeros((60000, 10), dtype='uint8')
    Y_test = np.zeros((10000, 10), dtype='uint8')

    for i, row in enumerate(Y):
        Y_train[i, y_train[i] -1] = 1
        Y_test[i, y_test[i] -1] = 1

    y = torch.tensor(y_train).type(torch.LongTensor)
    Y = torch.tensor(Y_train).type(torch.FloatTensor)
    y_test = torch.tensor(y_test).type(torch.LongTensor)
    Y_test = torch.tensor(Y_test).type(torch.FloatTensor)

    data = dataSet(X,y,(28,28,1),Y,X_test,y_test,Y_test)

    NN_ARCHITECTURE = [
        {"type": "CONV", "activation":"relu","filter_shape": (4,4),"no_channels":16,"stride":1,"padding":0},
        {"type": "CONV", "activation":"relu","filter_shape": (4,4),"no_channels":32,"stride":1,"padding":0},
        {"type": "POOL", "activation":"relu","filter_shape": (2,2),"stride":1,"padding":0},
        {"type": "CONV", "activation":"relu","filter_shape": (4,4),"no_channels":64,"stride":1,"padding":0},
        {"type":"DROPOUT","keep_prob":0.1},
        {"type": "CONV", "activation":"relu","filter_shape": (4,4),"no_channels":128,"stride":1,"padding":0},
        {"type": "Dense","activation":"tanh", "no_logits": 3200},
        {"type": "Dense","activation":"softmax", "no_logits": 10},
    ]

    q = CNN(NN_ARCHITECTURE,data)
    q.fit(0.00001,15)

