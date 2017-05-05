import numpy as np
from sklearn import datasets

from algorithms.logistic_regression import LogisticRegression

#use iris dataset with 2 features for testing
dataset = datasets.load_iris()

#load dataset with 2 features
X_full = dataset.data[:100,:2]
y_full = dataset.target[:100]

#add linear component feature
X_full = np.insert(X_full, 0, 1, axis=1)

#split datasets in 60:20:20
indexes_for_train = [i for i in sorted(range(0,100,2) + range(1,100,10))]
indexes_for_test = [i for i in range(0,100) if i not in indexes_for_train]

X_train = X_full[indexes_for_train,:]
y_train = y_full[indexes_for_train]

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


#plot decision boundery
algorithm.plot_decision_boundary(X_test,y_test)





