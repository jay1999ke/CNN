import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class Convolutional(object):

    def __init__(self,in_dimentions,filter_shape,no_channels,stride,padding,pre,activation):
        
        #imp info
        self.in_dimentions = in_dimentions
        self.filter_shape = filter_shape
        self.no_channels = no_channels
        self.stride = stride
        self.padding = padding
        self.prev = pre
        self.next=None
        self.activation = Activation(activation)

        #parameters
        self.filter = torch.tensor(0.09 * np.random.randn(no_channels,in_dimentions[2],filter_shape[0], filter_shape[0])).type(torch.FloatTensor).cuda()
        self.bias = torch.tensor(0.09 * np.random.randn(no_channels)).type(torch.FloatTensor).cuda()

        #activations calculations
        (n_h, n_w, n_c) = in_dimentions
        n_h = (n_h - filter_shape[0] + 2*padding)/stride + 1
        n_w = (n_w - filter_shape[1] + 2*padding)/stride + 1
        n_c = no_channels
        self.out_dimentions = (n_h, n_w,n_c)

        print("CONV", self.filter.size())

    def forward(self,input_block):

        convolutions = F.conv2d(input=input_block,
            weight=self.filter,
            bias=self.bias,
            stride=(self.stride,self.stride),
            padding = (self.padding,self.padding)
        )

        self.activations = self.activation.activate(convolutions)

        return self.activations


class Pooling(object):

    def __init__(self,in_dimentions,filter_shape,stride,padding,pre,type):
        
        #imp info
        self.in_dimentions = in_dimentions
        self.filter_shape = filter_shape
        self.no_channels = in_dimentions[2]
        self.stride = stride
        self.padding = padding
        self.prev = pre
        self.type = type
        self.next=None

        #activations calculations
        (n_h, n_w, n_c) = in_dimentions
        n_h = (n_h - filter_shape[0] + 2*padding)/stride + 1
        n_w = (n_w - filter_shape[1] + 2*padding)/stride + 1
        self.out_dimentions = (n_h, n_w,n_c)

        print("POOL", self.filter_shape)

    def forward(self,input_block):

        if self.type == "AVG":
            self.activations = F.avg_pool2d(input=input_block,
                kernal_size=self.filter_shape,
                stride=(self.stride,self.stride),
                padding = (self.padding,self.padding)
            )

        if self.type == "MAX":
            self.activations,self.indices = F.max_pool2d(input=input_block,
                kernal_size=self.filter_shape,
                stride=(self.stride,self.stride),
                padding = (self.padding,self.padding),
                return_indices=True
            )
        
        return self.activations

    def unpoolMask(self):
        """torch.nn.functional.max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None)"""
        """https://pytorch.org/docs/stable/nn.html#pooling-functions"""

        if self.type == "AVG":
            
            mask = torch.ones(self.in_dimentions[2],self.in_dimentions[3])/(self.filter_shape[0]*self.filter_shape[1])

        if self.type == "MAX":

            mask = F.max_unpool2d(input=self.activations,
                indices=self.indices,
                kernel_size=self.filter_shape,
                stride=self.stride,
                padding=self.padding
            )
            mask[mask !=0 ] = 1

        return mask

    def propagateError(self):
        pass

class Dense(object):

    def __init__(self,in_dimentions,no_logits,pre,activation):
        
        #imp info
        self.in_dimentions = in_dimentions
        self.no_logits = no_logits
        self.prev = pre
        self.next=None
        self.activation = Activation(activation)

        if isinstance(pre, Dense):
            in_n = pre.out_dimentions[0]
        else:
            (n_h, n_w, n_c) = pre.out_dimentions
            in_n = n_c*n_h*n_w

        #parameters
        self.theta = torch.tensor(0.09 * np.random.randn(int(in_n),int(no_logits))).type(torch.FloatTensor).cuda()
        self.bias = torch.tensor(0.09 * np.random.randn(no_logits,1)).type(torch.FloatTensor).cuda()
                

        #activations calculations
        self.out_dimentions = (in_n,no_logits)

        print("DENSE", self.theta.size())

    def forward(self,input_block):
        
        if not isinstance(self.prev, Dense):
            input_block = input_block.view(self.prev.out_dimentions[0],-1)
        self.Z = torch.mm(input_block,self.theta) + self.bias
        self.activations = self.activation.activate(self.Z)
        return self.activations



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









