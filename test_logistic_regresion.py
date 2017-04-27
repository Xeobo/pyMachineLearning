from logistic_regression import LogisticRegression
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import random

dataset = datasets.load_iris()

#load dataset with 2 features
X_full = dataset.data[:100,:2]
y_full = dataset.target[:100]

#add linear component feature
X_full = np.insert(X_full, 0, 1, axis=1)

#split datasets in 60:20:20
indexes_for_train = [i for i in sorted(range(0,100,2) + range(1,100,10))]
indexes_for_cv = sorted([i for i in range(0,100) if i not in indexes_for_train and i%10 <= 5])
indexes_for_test = sorted([i for i in range(0,100) if i not in indexes_for_train and i%10 > 5])

X_train = X_full[indexes_for_train,:]
y_train = y_full[indexes_for_train]

X_cross_validation = X_full[indexes_for_cv,:]
y_cross_validation = y_full[indexes_for_cv]

X_test = X_full[indexes_for_test,:]
y_test = y_full[indexes_for_test]


#it's ok to init with all zeros for logistic regression
init_theta = np.zeros(X_train.shape[1])

#train algotithm
algorithm = LogisticRegression(init_theta, X_train, y_train, 0.1, 0.1)
algorithm.minimize_with_gradient()

#calculate accuracy
accuracy = np.sum(np.array(y_train) == algorithm.predict(X_train)) / float(y_train.size)
print ("accuracy: " + str(accuracy))

#scatter positive and example
plt.scatter(X_test[:10, 1], X_test[:10, 2], c='b')
plt.scatter(X_test[10:, 1], X_test[10:, 2], c='r')

#plot decision boundery
X_plot = np.array([4, 7])

y_plot = -((X_plot - algorithm.mean[0]) / algorithm.deviation[0] * algorithm.get_theta()[1] + algorithm.get_theta()[0]) / algorithm.get_theta()[2] * algorithm.deviation[1] + algorithm.mean[1]

plt.plot(X_plot, y_plot, c='black')

plt.show()

# print("x = " + str(X_plot[0]) + ", y = " + str(y_plot[0]) + ", sigmoid = " + str(algorithm.sigmoid_function(np.array([1, X_plot[0], y_plot[0]]))))
#
# for row,rowy  in zip(X_train, y_train):
#     print("x1 = " + str(row) + ", sigmoid1 = " + str(algorithm.sigmoid_function(algorithm.normalize(np.array(row)))) + ", prediction: " + str(algorithm.predict(row)) + " ,rowy:" + str(rowy))
#
# print("predict all: " + str(algorithm.predict(X_train)))




