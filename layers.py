import numpy as np
import torch
import torch.nn.functional as F
from functional import Activation


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
        self.filter_grad = None
        self.activation = Activation(activation)

        #parameters
        self.filter = 0.09 *torch.randn(no_channels,in_dimentions[2],filter_shape[0], filter_shape[0]).cuda()
        self.bias = 0.09 * torch.randn(no_channels).cuda()

        #activations calculations
        (n_h, n_w, n_c) = in_dimentions
        n_h = int((n_h - filter_shape[0] + 2*padding)/stride + 1)
        n_w = int((n_w - filter_shape[1] + 2*padding)/stride + 1)
        n_c = int(no_channels)
        self.out_dimentions = (n_h, n_w,n_c)

        print("CONV", self.filter.size())

    def get(self,model):
        print(model.data.X.size())

    def forward(self,input_block):

        convolutions = F.conv2d(input=input_block,
            weight=self.filter,
            bias=self.bias,
            stride=(self.stride,self.stride),
            padding = (self.padding,self.padding)
        )

        self.activations = self.activation.activate(convolutions)

        return self.activations

    def backward(self,error):

        delta = error * self.activation.derivative(self.activations)	        
        self.bias_grad = delta.mean(0).mean(1).mean(1).view(self.no_channels)
        
        try:
            self.filter_grad = -1*F.conv2d(input = self.prev.activations.permute(1,0,2,3),
                weight = delta.permute(1,0,2,3),
                stride = (self.stride,self.stride),
                padding = (self.padding,self.padding)
            ).permute(1,0,2,3)

        except:
            self.filter_grad = -1*F.conv2d(input = self.prev.X.permute(1,0,2,3),
                weight = delta.permute(1,0,2,3),
                stride = (self.stride,self.stride),
                padding = (self.padding,self.padding)
            ).permute(1,0,2,3)


        #check for derivation of prev layer error when prev="CONV"
        error = F.conv_transpose2d(input = delta,
            weight = self.filter,
            stride=(self.stride,self.stride),
            padding = (self.padding,self.padding)
        )            

        return error

    def update(self,learning_rate):
        self.filter -= learning_rate*self.filter_grad
        self.bias -= learning_rate*self.bias_grad

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
        n_h = int((n_h - filter_shape[0] + 2*padding)/stride + 1)
        n_w = int((n_w - filter_shape[1] + 2*padding)/stride + 1)
        self.out_dimentions = (n_h, n_w,n_c)

        print("POOL", self.filter_shape)

    def forward(self,input_block):

        if self.type == "AVG":
            self.activations = F.avg_pool2d(input=input_block,
                stride=(self.stride,self.stride),
                kernel_size = self.filter_shape,
                padding = (self.padding,self.padding),
            )

        if self.type == "MAX":
            self.activations,self.indices = F.max_pool2d(input=input_block,
                kernel_size=self.filter_shape,
                stride=(self.stride,self.stride),
                padding = (self.padding,self.padding),
                return_indices=True,
            )
        
        return self.activations

    def unpoolMask(self):
        """torch.nn.functional.max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None)"""
        """https://pytorch.org/docs/stable/nn.html#pooling-functions"""

        if self.type == "AVG":
            
            mask = torch.ones(self.in_dimentions[2],
                self.in_dimentions[0],
                self.in_dimentions[1]
            )/(self.filter_shape[0]*self.filter_shape[1])

        if self.type == "MAX":

            mask = F.max_unpool2d(input=self.activations,
                indices=self.indices,
                kernel_size=self.filter_shape,
                stride=self.stride,
                padding=self.padding
            )
            mask[mask !=0 ] = 1

        return mask.cuda()

    def backward(self,error):

        ones_filter = torch.ones(self.in_dimentions[2],
            self.in_dimentions[2],
            self.filter_shape[0],
            self.filter_shape[1]
        ).cuda()
        error = F.conv_transpose2d(input=error,
            weight=ones_filter,
            stride=(self.stride,self.stride),
            padding = (self.padding,self.padding),
        )
        prev_layer_error = error * self.unpoolMask()
        return prev_layer_error

class Dense(object):

    def __init__(self,in_dimentions,no_logits,pre,activation):
        
        #imp info
        self.in_dimentions = in_dimentions
        self.no_logits = no_logits
        self.prev = pre
        self.next=None
        self.activation = Activation(activation)

        if isinstance(pre, Dense):
            in_n = pre.out_dimentions[1]
        else:
            try:
                (n_h, n_w, n_c) = pre.out_dimentions
            except:
                (n_h, n_w, n_c) = pre.img_dim
            in_n = n_c*n_h*n_w
        #parameters
        self.theta = 0.09 * torch.randn(int(in_n),int(no_logits)).cuda()
        self.bias = 0.09 * torch.randn(1,no_logits).cuda()
                

        #activations calculations
        self.out_dimentions = (in_n,no_logits)

        print("DENSE", self.theta.size())

    def forward(self,input_block):
        
        if not isinstance(self.prev, Dense):
            input_block = input_block.view(int(input_block.size()[0]),-1)
        self.Z = torch.mm(input_block,self.theta) + self.bias
        if self.activation.type == "softmax":
            self.activations = self.activation.activate(self.Z,1)
        else:
            self.activations = self.activation.activate(self.Z)
        return self.activations
        

    def backward(self,error):

        if(self.next != None):
            delta = error * self.activation.derivative(self.activations)
        else:
            delta=error

        self.bias_grad = delta.mean(0).view(1,self.no_logits)

        try:
            a = self.prev.in_dimentions
            first_layer = False
        except:
            first_layer = True

        if first_layer:
            prev_activations = self.prev.X.view(self.prev.X.size()[0],-1).transpose(1,0)
            self.theta_grad = -1*torch.mm(prev_activations,delta)

        elif isinstance(self.prev,Dense) or (isinstance(self.prev,Dropout) and isinstance(self.prev.prev,Dense)):
            self.theta_grad = -1*torch.mm(self.prev.activations.transpose(1,0),delta)

            return torch.mm(delta,self.theta.transpose(0,1))

        else:
            """ grad calculation"""	
            prev_activations = self.prev.activations.view(self.prev.activations.size()[0],-1).transpose(1,0)	
            self.theta_grad = -1*torch.mm(prev_activations,delta)	

            """prev layer error calculation"""	
            return torch.mm(delta,self.theta.transpose(1,0)).view(delta.size()[0],	
                int(self.in_dimentions[2]),	
                int(self.in_dimentions[0]),	
                int(self.in_dimentions[1])	
            )

    def update(self,learning_rate):
        self.theta -= learning_rate*self.theta_grad
        self.bias -= learning_rate*self.bias_grad

class Dropout(object):

    def __init__(self,in_dimentions,pre,keep_prob):
        
        #imp info
        self.in_dimentions = in_dimentions
        self.prev = pre
        self.next=None
        self.keep_prob = keep_prob

        #activations calculations
        self.out_dimentions = in_dimentions

        print("DROPOUT", self.keep_prob)

    def forward(self,input_block):
        if isinstance(self.prev, Dense):
            self.filter = torch.randn(self.in_dimentions[2],self.in_dimentions[0], self.in_dimentions[1]).cuda()
        else:
            self.filter = torch.randn(self.in_dimentions[0], self.in_dimentions[1]).cuda()
        self.filter[self.filter >= (1-self.keep_prob)] = 1
        self.filter[self.filter != 1] = 0
        self.activations = self.filter * input_block
        return self.activations


    def backward(self,error):

        prev_layer_error = self.filter * error

        return prev_layer_error
