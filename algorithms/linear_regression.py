import numpy as np
from algorithm import Algorithm
import matplotlib.pyplot as plt

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

        np_x = self.normalize(x)
        return np_x.dot(self.theta)

    def plot_data(self,X,y):
        #set min and max values
        x_min, x_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        #step
        h = 0.01
        #genetate points between min and max value
        xx = np.arange(x_min, x_max, h)
        #bios features
        ones = np.ones(xx.shape)
        #predict
        yy = self.predict(np.c_[ones.ravel(),xx.ravel()])

        plt.scatter(xx,yy, s=1)
        plt.scatter(X[:,1], y, s=40, c=y, cmap=plt.cm.Spectral)
        plt.show()
