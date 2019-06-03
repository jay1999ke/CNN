import numpy as np
import torch
import torch.nn.functional as F

class dataSet(object):
    def __init__(self, X, y,img_dim):
        self.X = X.cuda()
        self.y = y.cuda()

        #dims
        # (n_h, n_w,n_c)
        self.img_dim = img_dim

class Activation(object):

    def __init__(self,type):
        self.type = type.lower()

    def activate(self,data):

        type = self.type

        if type == "tanh":
            return torch.tanh(data)
        elif type == "sigmoid":
            return torch.sigmoid(data)
        elif type == "relu":
            return torch.relu(data)

    def derivative(self,data):

        type = self.type

        if type == "tanh":
            tanh = torch.tanh(data)
            return 1 - tanh*tanh
        elif type == "sigmoid":
            sig = torch.sigmoid(data)
            return sig*(1-sig)
        elif type == "relu":
            data[data <= 0] = 0
            data[data > 0] = 1
            return data

class Cost(object):

    def __init__(self,type):
        self.type = type.lower()

    def calculate(self,input,target):

        if self.type == "cross_entropy":
            return F.cross_entropy(input=input, target=target)
