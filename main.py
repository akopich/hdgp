from HDGPRegression import HDGRegression
import numpy as np


X = np.array([[0, 10], [1, -20], [2, 30]])
y = np.array([1, 2, 3]).reshape(-1, 1)

r = HDGRegression(rho=1, sigma2=0.2)
r.fit(X, y)
print(r.result)