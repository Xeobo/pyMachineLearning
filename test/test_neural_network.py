import numpy as np
from sklearn import datasets
from algorithms.neural_network import NeuralNetwork

HIDDEN_LAYER_SIZE = 4

# Generate a dataset and plot it
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)

#insert linear component
X = np.insert(X,0,1,axis=1)

#init theta with random theta
np.random.seed(0)
W1 = np.random.randn(np.array(X).shape[1], HIDDEN_LAYER_SIZE) / np.sqrt(np.array(X).shape[1])
W2 = np.random.randn(HIDDEN_LAYER_SIZE + 1, 1) / np.sqrt(HIDDEN_LAYER_SIZE + 1)
theta = [W1.T.tolist()] + [W2.T.tolist()]


#minimize algorithm
algotithm = NeuralNetwork(theta, X, y, 16, 0.1)
algotithm.minimize_with_gradient()

#plot
algotithm.plot_decision_boundary(X,y)











