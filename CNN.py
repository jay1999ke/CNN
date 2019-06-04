import numpy as np
import torch
import torch.nn.functional as op
from layers import Convolutional, Pooling, Dense
from functional import Activation,Cost


class CNN(object):

    def __init__(self,architecture,data):
        self.architecture = architecture
        self.data = data
        self.model = self.createModel()

    def createModel(self):
        print("\n")
        model = []
        dimentions = self.data.img_dim
        pre=None

        for i,layer in enumerate(self.architecture):

            l_type = layer["type"]

            if l_type == "CONV":
                conv = Convolutional(in_dimentions=dimentions,
                    filter_shape=layer["filter_shape"],
                    no_channels=layer["no_channels"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    pre=pre,
                    activation = layer["activation"]
                )

                if i != 0:
                    pre.next = conv
                    
                dimentions = conv.out_dimentions
                pre = conv
                model.append(conv)

            elif l_type == "Dense":
                dense = Dense(in_dimentions=dimentions,
                    no_logits=layer["no_logits"],
                    pre=pre,
                    activation = layer["activation"]
                )

                if i != 0:
                    pre.next = dense

                dimentions = dense.out_dimentions
                pre = dense
                model.append(dense)

            elif l_type == "POOL":
                pool = Pooling(in_dimentions=dimentions,
                    filter_shape=layer["filter_shape"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    pre=pre,
                    type=layer["p_type"]
                )

                if i != 0:
                    pre.next = pool
                    
                dimentions = pool.out_dimentions
                pre = pool
                model.append(pool)

        print("\n")
                
        return model

    def forward(self):
        input_block = self.data.X

        for i,layer in enumerate(self.model):       

            input_block = layer.forward(input_block)
            print("Layer ", i)    

        self.activations = input_block
        return self.activations

    def cost(self):
        cross_entropy = Cost("cross_entropy")

        return cross_entropy.calculate(self.activations,self.data.y)

    def backward(self):
        
        error = self.activations - self.data.y_one_hot

        list_len = len(self.model)-1
        for i,layer in enumerate(self.model[::-1]):
            i=list_len-i

            error = layer.backward(error)

            print("Layer ", i,error.size())