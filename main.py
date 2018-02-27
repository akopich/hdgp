from HDGPRegression import HDGRegression
import numpy as np
import scipy
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor

X = np.array([[0, 0], [1, 1], [2, 2]])
y = np.array([1, 2, 3]).reshape(-1, 1)

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target.reshape(-1, 1)
X = preprocessing.scale(X)

r = HDGRegression(rho=1e100)

print(y.var())
print(cross_val_score(r, X, y, cv = 4, scoring='neg_mean_squared_error', n_jobs=4))
lr = GaussianProcessRegressor()
print(cross_val_score(lr, X, y, cv = 4, scoring='neg_mean_squared_error'))

# print(r.fit(X,y))
# xtest = X[0:2, :]
# print(r.predict(xtest))