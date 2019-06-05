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
        pre=self.data

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

        self.activations = input_block
        return self.activations

    def forward_test(self):
        input_block = self.data.X_test

        for i,layer in enumerate(self.model):       

            input_block = layer.forward(input_block)

        self.activations_test = input_block
        return self.activations_test

    def cost(self):

        return self.get_cost_value(self.activations,self.data.y_one_hot)

    def backward(self):
        A = self.activations
        y = self.data.y_one_hot
        error = y - A

        list_len = len(self.model)-1
        for i,layer in enumerate(self.model[::-1]):
            i=list_len-i

            error = layer.backward(error)

    def update(self):

        for i,layer in enumerate(self.model[::-1]):

            if not isinstance(layer,Pooling):
                layer.update(self.alpha)

    def fit(self,learning_rate,epochs):
        self.alpha = learning_rate
        self.epochs = epochs
        print("iter","cost","train_acc","train_pred","test_acc", "test_pred"  )
        for i in range(epochs):
            self.forward()
            self.backward()
            self.update()


            if(i%10==0):
                self.forward_test()
                print(i,self.cost(),
                    self.get_accuracy_value(self.activations,self.data.y_one_hot),
                    self.get_accuracy_value(self.activations_test,self.data.y_test_one_hot),           
                    self.accuracy_quick(self.activations,self.data.y),
                    self.accuracy_quick(self.activations_test,self.data.y_test)
                
                )

    def get_accuracy_value(self,Y_hat, Y):
        Y_hat_ = self.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(dim=1).type(torch.FloatTensor).mean()

    def convert_prob_into_class(self,probs):
        probs_ = probs.clone()
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_

    def get_cost_value(self,Y_hat, Y):
        # number of examples
        m = Y_hat.shape[1]
        # calculation of the cost according to the formula
        cost = -1 / m * (Y* torch.log(Y_hat) + (1 - Y)* torch.log(1 - Y_hat))
        #cost = np.sum((1 / 2) * (Y - Y_hat) * (Y - Y_hat))
        return torch.sum(torch.squeeze(cost))

    def accuracy_quick(self, activation, y):
        # calling code must set mode = 'train' or 'eval'
        (max_vals, arg_maxs) = torch.max(activation, dim=1) 
        # arg_maxs is tensor of indices [0, 1, 0, 2, 1, 1 . . ]
        arg_maxs = arg_maxs 

        num_correct = torch.sum(y==(arg_maxs+1)).type(torch.FloatTensor)
        acc = num_correct /y.size()[0]
        return acc  # percentage based


        