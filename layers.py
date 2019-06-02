import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class Convolutional(object):

    def __init__(self,in_dimentions,filter_shape,no_channels,stride,padding,pre):
        
        #imp info
        self.in_dimentions = in_dimentions
        self.filter_shape = filter_shape
        self.no_channels = no_channels
        self.stride = stride
        self.padding = padding
        self.prev = pre
        self.next=None

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

    def conv_forward(self,input_block):

        self.activations = F.conv2d(input=input_block,
            weight=self.filter,
            bias=self.bias,
            stride=(self.stride,self.stride),
            padding = (self.padding,self.padding)
        )




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

    def pool_forward(self,input_block):

        if self.type == "AVG":
            self.activations = F.avg_pool2d(input=input_block,
                kernal_size=self.filter_shape,
                stride=(self.stride,self.stride),
                padding = (self.padding,self.padding)
            )

        if self.type == "MAX":
            self.activations = F.max_pool2d(input=input_block,
                kernal_size=self.filter_shape,
                stride=(self.stride,self.stride),
                padding = (self.padding,self.padding)
            )




class Dense(object):

    def __init__(self,in_dimentions,no_logits,pre):
        
        #imp info
        self.in_dimentions = in_dimentions
        self.no_logits = no_logits
        self.prev = pre
        self.next=None

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








