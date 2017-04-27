import numpy as np
from algorithm import Algorithm

class LogisticRegression(Algorithm):

    def __init__(self,initTheta, data,labels, alpha_param,regularisation_param = 0):
        Algorithm.__init__(self,initTheta,data,labels, alpha_param, regularisation_param)

    def _cost_derivate(self):
        derivate = np.transpose((self._sigmoid_function() - self.labels)).dot(self.data)/self.data[:,0].size

        return derivate

    def _sigmoid_function(self):
        return np.vectorize(lambda x: 1.0/(1.0+ np.exp(-x)))(self.data.dot(self.theta))

    def sigmoid_function(self,data):
        return np.vectorize(lambda x: 1.0/(1.0+ np.exp(-x)))(data.dot(self.theta))

    def cost_function(self):
        prediction = self._sigmoid_function()

        return (-self.labels.dot(np.log(prediction)) - (1 -self.labels).dot(np.log(1- prediction)) + self.lambda_param *self.theta.dot(self.theta)/2)/self.data[:,0].size

    def predict(self,x):

        np_x = self.normalize(x)

        return np.array(self.sigmoid_function(np_x) >= 0.5)

    def get_theta(self):
        return self.theta
