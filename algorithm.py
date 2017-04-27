import numpy as np
from abc import ABCMeta,abstractmethod
from time import *
import math

#Maximum iterations of gradient Descent
MAX_ITERATION = 1000
CONVERGE_RATIO = 0.001

class Algorithm(object):
    __metaclass__ = ABCMeta
    def __init__(self,initTheta,data,labels,alpha_param,lambda_param):

        self.theta = np.array(initTheta)
        self.alpha_param = alpha_param
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.is_model_fit = False
        self.lambda_param = lambda_param

    @abstractmethod
    def _cost_derivate(self):
        pass

    @abstractmethod
    def cost_function(self):
        pass

    @abstractmethod
    def predict(self,x):
        pass

    @abstractmethod
    def predict(self,x):
        pass

    def normalize(self,x):
        np_x = np.array(x)
        if self.is_model_fit:
            if np_x.ndim > 1:
                np_x[:,1:] = (np.array(np_x[:,1:]) - self.mean)/self.deviation
            else:
                np_x[1:] = (np.array(np_x[1:]) - self.mean)/self.deviation
        else:
            raise self
        return np_x

    def get_theta(self):
        return self.theta

    def minimize_with_gradient(self):
        start = time()

        #normalize
        self.mean = np.mean(self.data[:,1:], axis=0)
        self.deviation = np.std(self.data[:,1:], axis=0)
        self.data[:,1:] = (self.data[:,1:] - np.mean(self.data[:,1:], axis=0)) / np.std(self.data[:,1:], axis=0)

        new_cost = self.cost_function()
        for i in range(0,MAX_ITERATION):
            old_cost = new_cost

            self.theta = self.theta - (self.alpha_param*self._cost_derivate())

            new_cost = self.cost_function()

            print("Current cost: " + str(new_cost) + ", old cost:" + str(old_cost) + ", abs: " + str(abs(old_cost - new_cost)))

            if abs(old_cost - new_cost) < CONVERGE_RATIO:
                break

        self.is_model_fit = True
        print("=================================")
        print("Algorithm minimized with cost: " + str(new_cost) + "in " + str(i) + " steps.")
        print("Minimized theta: " + str(self.theta))
        print("=================================")

        return


