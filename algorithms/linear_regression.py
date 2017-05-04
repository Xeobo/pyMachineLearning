import numpy as np
from algorithm import Algorithm
import math

class LinearRegression(Algorithm):

    def __init__(self,initTheta,data,labels,alpha_param,lambda_param = 0):
        Algorithm.__init__(self,initTheta,data,labels,alpha_param,lambda_param)

    def _linear_function(self):
        return self.data.dot(self.theta)

    def _cost_derivate(self):
        derivate = (np.transpose((self._linear_function() - self.labels)).dot(self.data) + self.lambda_param *self.theta)/self.data[:,0].size

        return derivate

    def cost_function(self):
        singular_cost = (self.data.dot(self.theta) - self.labels)

        return (singular_cost.dot(singular_cost) + self.lambda_param * self.theta.dot(self.theta)/2)/self.data[:,0].size


    def predict(self,x):
        np_x = np.array(x)
        np_x[:,1:] =  (np_x[:,1:] - self.mean)/ self.deviation
        return np_x.dot(self.theta)

