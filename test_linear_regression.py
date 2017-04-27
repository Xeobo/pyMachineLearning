import numpy as np
from linear_regression import LinearRegression

array1 = np.array([1,2,3])

array2 = np.array([4,7,8])

x = [1,2,3] + [4,6,8]

print(np.array((array1.tolist() + array2.tolist())))

print(x)

print(array1.dot(array2))

x = np.array([
    [1,4],
    [2,2],
    [1,3],
    [3,2]
])

y= np.array([5,4,4,5])

theta = np.array([0.001, 0.001])

algorithm = LinearRegression(theta, 0.1, x, y)

algorithm.minimize_with_gradient()

print("Cost function: " + str(algorithm.cost_function()))

print("Prediction function: " + str(algorithm.predict(x)))



