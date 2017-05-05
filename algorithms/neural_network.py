import numpy as np
from classification_algorithm import ClassificationAlgorithm
import math


class NeuralNetwork(ClassificationAlgorithm):
    def __init__(self, initTheta, data, labels, alpha_param, regularisation_param=0):
        labels = np.array([labels]).T
        ClassificationAlgorithm.__init__(self, initTheta, data, labels, alpha_param, regularisation_param)

    def _calculate_new_theta(self):
        """
        Because of theta being an array of matrices in Neural Network's perspective, calculating new values is
        different from general behaviour
        :return: new theta value calculated as a single step of gradient descent algorithm
        """
        new_theta = []
        cost_derivate = self._cost_derivate()


        for i in range(0, len(self.theta)):
            new_theta = new_theta + [self.theta[i] - (self.alpha_param * cost_derivate[i])]

        return np.array(new_theta)

    def sigmoid_base(self,X):
        return 1.0 / (1.0 + math.e ** (-X))

    def _sigmoid_gradient(self, x):
        """
        :param x: data on witch sigmoid gradient should be calculated
        :return: sigmoid gradient for data
        """
        zig = self.sigmoid_base(x)
        return zig * (1 - zig)

    def _cost_derivate(self):
        #do forward propagation
        [prediction, activations] = self._sigmoid_function_with_activations(self.data)
        #delta of the output layer
        delta = prediction - self.labels
        delta = [delta]
        #theta_gradients
        theta_grad = []

        #calculate delta errors for all layers except input layer, in reversed order
        for i in reversed(range(0, len(self.theta) - 1)):
            #delta errors from bios features shouldn't be accounted
            sum = delta[0].dot(np.array(self.theta[i + 1]))[:, 1:]
            #calculate delta error for current theta and append it to the top
            delta = [sum * self._sigmoid_gradient( activations[i].dot(np.array(self.theta[i]).T) )] + delta

        for i in range(0, len(self.theta)):

            theta_np = np.array(self.theta[i])
            #bios features shouldn't be considered in regularisation, so we set first column to zeros
            #switch case when theta is only one row matrix
            if theta_np.ndim > 1:
                theta_zero_first_column = theta_np[:, 1:]
                theta_zero_first_column = np.insert(theta_zero_first_column, 0, 0, axis=1)
            else:
                theta_zero_first_column = [0]
                theta_zero_first_column += theta_np[1:].tolist()

            theta_grad = theta_grad + [(1. / self.number_of_features) * (
                        delta[i].T.dot(activations[i]) + self.lambda_param * np.array(theta_zero_first_column))]

        return theta_grad

    def _sigmoid_function(self):
        return self._sigmoid_function_with_activations(self.data)[0]

    def _sigmoid_function_with_activations(self, data):
        """
        :return: in difference to logistic_regression sigmoid function returns activations functions for each layer
                as well as the predicted values from output layer
        """
        output = data
        activations = []

        for i in range(0, len(self.theta)):
            activations += [output]
            output = self.sigmoid_base(np.array(output).dot(np.array(self.theta[i]).T))

            #for all layers, except output layer add bios column
            if i != len(self.theta) - 1:
                output = np.insert(output, 0, 1, axis=1).tolist()

        return [output, activations]

    def prediction_probability(self, x):
        return self._sigmoid_function_with_activations(x)[0]

    def cost_function(self):
        #calculate predictions
        prediction = self._sigmoid_function()
        regularisation = 0

        #for each theta calculate regularisation
        for i in range(0, len(self.theta)):
            theta_np = np.array(self.theta[i])
            if theta_np.ndim > 1:
                regularisation += self.lambda_param * np.sum(theta_np[:, 1:] * theta_np[:, 1:])
            else:
                regularisation += self.lambda_param * np.sum(theta_np[1:] * theta_np[1:])

        #calculate cost function
        return (-self.labels.T.dot(np.log(prediction)) - (1 - self.labels).T.dot(
            np.log(1 - prediction)) + regularisation / 2) / self.data[:, 0].size
