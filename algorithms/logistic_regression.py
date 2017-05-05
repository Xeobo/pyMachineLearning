import numpy as np
import math
from classification_algorithm import ClassificationAlgorithm

class LogisticRegression(ClassificationAlgorithm):

    def __init__(self,initTheta, data,labels, alpha_param=0.1,regularisation_param = 0):
        ClassificationAlgorithm.__init__(self,initTheta,data,labels, alpha_param, regularisation_param)

    def _cost_derivate(self):
        #calculate cost derivate
        derivate = (self._sigmoid_function() - self.labels).dot(self.data)/self.number_of_features

        return derivate

    def _sigmoid_function(self):
        return self.prediction_probability(self.data)

    def prediction_probability(self, data):
        return 1.0/(1.0+ math.e**(-data.dot(self.theta.T)))

    def cost_function(self):
        prediction = self._sigmoid_function()

        print("shapes:" + str(self.labels.shape) + ", " + str(np.array(prediction).shape) + ", " + str(self.theta.shape))
        return (-self.labels.dot(np.log(prediction))
                - (1 -self.labels).dot(np.log(1- prediction))
                + self.lambda_param * np.sum(self.theta[1:] * self.theta[1:])/2)/self.number_of_features


