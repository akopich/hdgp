from HDGPRegression import HDGRegression
import numpy as np
import scipy
from sklearn import datasets

r = HDGRegression(1, 0.2)

X = np.array([[0, 0], [1, 1], [2, 2]])
y = np.array([1, 2, 3]).reshape(-1, 1)

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target.reshape(-1, 1)

print(r.fit(X,y))
print(r.B)