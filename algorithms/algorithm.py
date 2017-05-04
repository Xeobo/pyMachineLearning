import numpy as np
from abc import ABCMeta,abstractmethod
from time import *
import math

#Maximum iterations of gradient Descent
MAX_ITERATION = 10000
CONVERGE_RATIO = 10**-7

class Algorithm(object):
    __metaclass__ = ABCMeta

    def _normalize(self,x):
        return (np.array(x) - self.mean)/self.deviation

    def _calculate_new_theta(self):
        return self.theta - (self.alpha_param*self._cost_derivate())

    def __init__(self,initTheta,data,labels,alpha_param,lambda_param):
        """

        :param initTheta: initial value of theta parameters before minimisation. Some algorithms demand special start
                values for minimisation to be successful (ex. Neural networks)
        :param data: training data to be used in minimisation process.
        :param labels: values needed by supervised learning algorithm to learn on.
        :param alpha_param: parameter used by gradient descent algorithm for fitting data. If too big large algorithm
                may diverge, if too small, algorithm may take to small time to converge, or may converge to wrong value
        :param lambda_param: used for regularisation to prevent overfitting the dataset. If no regularisation is wanted
                parameter should be zero.
        """
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
        """
        Used to normalize data set in similar ranges. User <b>shouldn't</b> call this function before minimizing
        operation. Use function for other calculations, such as calculating decision boundary for classification
        algorithms. <b>Function should be called only after minimizing data.</b>

        :param x: input data which want's to bee normalized.
        :return: normalazed data
        """
        np_x = np.array(x)
        if self.is_model_fit:
            if np_x.ndim > 1:
                np_x[:,1:] = self._normalize(np_x[:,1:])
            else:
                np_x[1:] = self._normalize(np_x[1:])
        else:
            raise NotImplementedError("Not gonna happen bro")
        return np_x

    def get_theta(self):
        return self.theta

    def minimize_with_gradient(self):
        start = time()

        #normalize
        self.mean = np.mean(self.data[:,1:], axis=0)
        self.deviation = np.std(self.data[:,1:], axis=0)

        self.is_model_fit = True

        self.data[:,1:] = self._normalize(self.data[:,1:])

        new_cost = self.cost_function()
        for i in range(0,MAX_ITERATION):
            old_cost = new_cost

            self.theta = self._calculate_new_theta()

            new_cost = self.cost_function()

            if i%100:
                print("Current cost: " + str(new_cost) + ", old cost:" + str(old_cost) + ", abs: " + str(abs(old_cost - new_cost)))

            if abs(old_cost - new_cost) < CONVERGE_RATIO:
                break

        print("=================================")
        print("Algorithm minimized with cost: " + str(new_cost) + "in " + str(i) + " steps, in " + str(time() - start) + " s." )
        print("Minimized theta: " + str(self.theta))
        print("=================================")

        return


