import numpy as np
from algorithm import Algorithm
import math
class NeuralNetwork(Algorithm):

    def __init__(self,initTheta, data,labels, alpha_param,regularisation_param = 0):
        self.sigmoid_base = np.vectorize(lambda x: 1.0/(1.0+ math.e**(-x)))
        Algorithm.__init__(self,initTheta,data,labels, alpha_param, regularisation_param)

    def _calculate_new_theta(self):
        new_theta = []
        cost_derivate = self._cost_derivate()
        for i in range(0,self.theta.shape[0]):
            new_theta = new_theta + [self.theta[i] - (self.alpha_param*cost_derivate[i])]


        return np.array(new_theta)

    def _sigmoid_gradient(self,x):
        zig = self.sigmoid_base(x)
        return zig*(1-zig)


    def _cost_derivate(self):
        [prediction, activations] = self.sigmoid_function(self.data)

        delta = prediction - self.labels

        delta = [delta.tolist()]

        theta_grad = []

        for i in reversed(range(0,self.theta.shape[0]-1)):
            sum = (np.array(delta[0])).dot(np.array(self.theta[i+1]))[:,1:]
            delta = [(sum*self._sigmoid_gradient(activations[i].dot(np.array(self.theta[i]).T))).tolist()] + delta

        for i in range(0,self.theta.shape[0]):

            theta_np =np.array(self.theta[i])
            if theta_np.ndim > 1:
                theta_zero_first_column = theta_np[:,1:]
                theta_zero_first_column = np.insert(theta_zero_first_column,0,0,axis=1)
            else:
                theta_zero_first_column = [0]
                theta_zero_first_column += theta_np[1:].tolist()


            theta_grad = theta_grad + [(1./self.data[:,0].size)*( np.array(delta[i]).T.dot(activations[i]) +  self.lambda_param * np.array(theta_zero_first_column))]

        return theta_grad



    def _sigmoid_function(self):

        return self.sigmoid_function(self.data)[0]

    def sigmoid_function(self,data):
        output = data
        activations = []
        for i in range(0,self.theta.shape[0]):
            activations += [output]
            output = self.sigmoid_base(np.array(output).dot(np.array(self.theta[i]).T))

            if i != self.theta.shape[0] - 1:
                output = np.insert(output,0,1,axis=1).tolist()

        return [output,activations]

    def cost_function(self):
        prediction = self._sigmoid_function()
        regularisation = 0

        for i in range(0,self.theta.shape[0]):
            theta_np = np.array(self.theta[i])
            if theta_np.ndim>1:
                regularisation += self.lambda_param*np.sum(theta_np[:,1:]*theta_np[:,1:])
            else:
                regularisation += self.lambda_param*np.sum(theta_np[1:]*theta_np[1:])

        return (-self.labels.T.dot(np.log(prediction)) - (1 -self.labels).T.dot(np.log(1- prediction)) + regularisation/2)/self.data[:,0].size

    def predict(self,x):
        np_x = x

        np_x[:,1:] = self._normalize(x[:,1:])

        return np.array(self.sigmoid_function(np.array(x))[0] >= 0.5)


