#Author : github.com/jay1999ke
"""
neural net
"""
import numpy as np
import matplotlib.pyplot as plt
from csvr import datareader
import scipy.io as mat
import torch

NN_ARCHITECTURE = [
    {"input_dim": 401, "output_dim": 200},
    {"input_dim": 200, "output_dim": 100},
    {"input_dim": 100, "output_dim": 10}
]


class dataSet(object):
    def __init__(self, X, y):
        self.X = X.cuda()
        self.y = y.cuda()


class neuralNet(object):

    def __init__(self, data, architecture, alpha, epochs):
        self.epochs = epochs
        self.alpha = alpha
        self.data = data
        self.architecture = architecture
        self.theta = self.createTheta()
        self.grad = {}
        self.activations = {}
        self.predictions = data.y
        self.delta = {}

        for x in self.theta:
            print(x, self.theta[x].shape)

    def sigmoid(self, Z):
        exp = torch.exp(-Z)
        return 1/(1+exp)

    def der_sigmoid(self, Z):
        sig = self.sigmoid(Z)
        return sig * (1 - sig)

    def createTheta(self):
        seed = 98
        np.random.seed(seed)

        theta = {}

        for i, layer in enumerate(self.architecture):
            # we number network layers from 1
            layer_i = i + 1

            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            theta['theta' + str(layer_i)] = torch.tensor(0.09 * np.random.randn(layer_input_size, layer_output_size)).type(torch.FloatTensor).cuda()

        return theta

    def single_layer_forward(self, A_prev, theta_curr):
        Z_curr = torch.mm(A_prev, theta_curr)
        return self.sigmoid(Z_curr), Z_curr

    def forward(self):
        # X vector is the activation for layer 0 
        A_curr = self.data.X

        for i in range(len(self.architecture)):
            layer_i = i + 1
            # transfer the activation from the previous iteration
            A_prev = A_curr

            W_curr = self.theta["theta" + str(layer_i)]
            A_curr, Z_curr = self.single_layer_forward(A_prev, W_curr)

            self.activations["A" + str(i)] = A_prev
            self.activations["A" + str(i)].transpose(0,1)[0] = 1
            self.activations["Z" + str(layer_i)] = Z_curr

        self.activations['A'+str(layer_i)] = A_curr
        self.predictions = A_curr

    def cost(self):
        y = self.data.y
        h = self.predictions
        return get_cost_value(h,y)/10

    def delta_calculater(self):

        an = self.predictions
        self.delta["d"+str(len(self.architecture))] =  self.data.y - an
        
        for layer_i in range(len(self.architecture)-1,0,-1):

            theta_curr = self.theta["theta"+str(layer_i + 1)]
            activation_err = torch.mm(self.delta["d"+str(layer_i + 1)],theta_curr.transpose(0,1))

            d_sig = self.der_sigmoid(self.activations["A"+str(layer_i)]).transpose(0,1)

            self.delta["d"+str(layer_i)] = activation_err * d_sig.transpose(0,1)

    def gradients(self):

        for i in range(1, len(self.architecture)+1):
            self.grad["grad" + str(i)] = -1* torch.mm(self.activations["A" + str(i-1)].transpose(0,1),self.delta["d"+str(i)]) + 1/5000*self.theta["theta"+str(i)]*self.theta["theta"+str(i)]

    def update_theta(self):

        for i in range(1, len(self.architecture)+1):
            self.theta["theta"+str(i)] -= self.alpha*self.grad["grad"+str(i)]

    def train(self):
        costs = []
        acc = []
        y=[]
        for i in range(self.epochs):
            self.forward()
            self.delta_calculater()
            self.gradients()
            self.update_theta()

            if(i % 50 == 0):
                c = get_cost_value(self.predictions, self.data.y)
                a = get_accuracy_value(self.predictions, self.data.y)
                costs.append(c)
                acc.append(a)
                y.append(i)
                print("iter: ", i, "cost: ", c,"Acc: ",a)
        
        fig, ax = plt.subplots(nrows=1, ncols=2)
        c=0
        X = [costs,acc]
        for col in ax:
            col.plot(y,X[c])
            c+=1

        plt.show()


def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (Y* torch.log(Y_hat) + (1 - Y)* torch.log(1 - Y_hat))
    #cost = np.sum((1 / 2) * (Y - Y_hat) * (Y - Y_hat))
    return torch.sum(torch.squeeze(cost))


def convert_prob_into_class(probs):
    probs_ = probs.clone()
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(dim=1).type(torch.FloatTensor).mean()


if __name__ == "__main__":

    raw_data = mat.loadmat("mnist_reduced.mat")
    #dataset
    X = raw_data['X']
    
    fig, ax = plt.subplots(nrows=2, ncols=5)
    c=0
    for row in ax:
        for col in row:
            col.imshow(X[c].reshape((20, 20)).T / 255.0)
            c+=500

    plt.show()

    X = np.append(np.ones((5000, 1)), X, axis=1).astype(np.float32)
    y = raw_data['y'].ravel()

    Y = np.zeros((5000, 10), dtype='uint8')

    for i, row in enumerate(Y):
        Y[i, y[i] - 1] = 1

    y = Y.astype(np.float32)

    

    X = torch.tensor(X)
    y = torch.tensor(y)

    data = dataSet(X, y)


    #net instance,forward test
    print("\ninit:")
    q = neuralNet(data, NN_ARCHITECTURE, 0.0001, 1200)
    q.train()



