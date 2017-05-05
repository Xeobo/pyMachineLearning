from algorithm import Algorithm
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod

class ClassificationAlgorithm(Algorithm):

    def __init__(self,initTheta, data,labels, alpha_param=0.1,regularisation_param = 0):
        Algorithm.__init__(self,initTheta,data,labels, alpha_param, regularisation_param)


    @abstractmethod
    def prediction_probability(self, x):
        """
        :param x: data to calculate prediction
        :return: returns predictions as row numbers in range 0 to 1.
        """
        pass

    def predict(self,x):
        #normalize features before calculating
        np_x = self.normalize(x)

        return np.array(self.prediction_probability(np_x) >= 0.5)

    def plot_decision_boundary(self,X,y,first_dimension=1,second_dimension=2):
        # Set min and max values and give it some padding
        x_min, x_max = X[:, first_dimension].min() - .5, X[:, first_dimension].max() + .5
        y_min, y_max = X[:, second_dimension].min() - .5, X[:, second_dimension].max() + .5
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        #bios features
        ones = np.ones(xx.shape)
        # Predict the function value for the whole grid
        Z = self.predict(np.c_[ones.ravel(),xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:,1], X[:,2], s=40, c=y, cmap=plt.cm.Spectral)
        plt.show()
