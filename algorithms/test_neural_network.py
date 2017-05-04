import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from algorithms.neural_network import NeuralNetwork

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    y_min, y_max = X[:, 2].min() - .5, X[:, 2].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    ones = np.ones(xx.shape)
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[ones.ravel(),xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,1], X[:,2], s=40, c=y, cmap=plt.cm.Spectral)


HIDDEN_LAYER_SIZE = 4

# Generate a dataset and plot it
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)

#insert linear component
X = np.insert(X,0,1,axis=1)
y = np.transpose([y])

np.random.seed(0)
W1 = np.random.randn(np.array(X).shape[1], HIDDEN_LAYER_SIZE) / np.sqrt(np.array(X).shape[1])
W2 = np.random.randn(HIDDEN_LAYER_SIZE + 1, 1) / np.sqrt(HIDDEN_LAYER_SIZE + 1)

theta = [W1.T.tolist()] + [W2.T.tolist()]

algotithm = NeuralNetwork(theta, X, y, 8, 0.1)
algotithm.minimize_with_gradient()

print( "cost function: "+ str(algotithm.cost_function()))

plot_decision_boundary(algotithm.predict)
plt.show()










