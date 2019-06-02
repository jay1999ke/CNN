import numpy as np
import matplotlib.pyplot as plt
import scipy.io as mat
import torch
import torch.nn.functional as op
from layers import Convolutional, Pooling, Dense

class dataSet(object):
    def __init__(self, X, y,img_dim):
        self.X = X.cuda()
        self.y = y.cuda()

        #dims
        # (n_h, n_w,n_c)
        self.img_dim = img_dim


class CNN(object):

    def __init__(self,architecture,data):
        self.architecture = architecture
        self.data = data
        self.model = self.createModel()

    def createModel(self):
        model = []
        #dimentions = self.data.img_dim
        pre=None

        dimentions = (28,28,1)

        for i,layer in enumerate(self.architecture):

            l_type = layer["type"]

            if l_type == "CONV":
                conv = Convolutional(in_dimentions=dimentions,
                    filter_shape=layer["filter_shape"],
                    no_channels=layer["no_channels"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    pre=pre
                )

                if i != 0:
                    pre.next = conv
                    
                dimentions = conv.out_dimentions
                pre = conv
                model.append(conv)

            elif l_type == "Dense":
                dense = Dense(in_dimentions=dimentions,
                    no_logits=layer["no_logits"],
                    pre=pre
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
                
        return model




if __name__ == "__main__":
    

    NN_ARCHITECTURE = [
        {"type": "CONV", "filter_shape": (5,5),"no_channels":6,"stride":1,"padding":0},
        {"type": "POOL","p_type":"AVG", "filter_shape": (2,2),"stride":2,"padding":0},
        {"type": "CONV", "filter_shape": (5,5),"no_channels":12,"stride":1,"padding":0},
        {"type": "POOL","p_type":"AVG",  "filter_shape": (2,2),"stride":2,"padding":0},
        {"type": "Dense", "no_logits": 10},
    ]

    q = CNN(NN_ARCHITECTURE,0)