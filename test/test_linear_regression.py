import numpy as np
import matplotlib.pyplot as plt
from algorithms.linear_regression import LinearRegression

#input data
x = np.array([
    [1,4],
    [1,2],
    [1,3],
    [1,5]
])
#output represents summation
y= np.array([5,3,4,6])

#initial theta values
theta = np.array([0.001, 0.001])

#minimize theta values
algorithm = LinearRegression(theta, x, y, 1)
algorithm.minimize_with_gradient()

print("Prediction function: " + str(algorithm.predict(x)))

algorithm.plot_data(x,y)



