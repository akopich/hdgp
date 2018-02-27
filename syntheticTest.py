from HDGPRegression import HDGRegression
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression

X = np.random.normal(0, 1, 10000).reshape(-1, 100)
y = np.sin(X[:, 0]).reshape(-1, 1)

crossval = lambda regressor: cross_val_score(regressor, X, y, cv=10, scoring='neg_mean_squared_error', n_jobs=4).mean()

r = HDGRegression(rho=0)
print("HDGR(rho=0): " + str(crossval(r)))
r = HDGRegression(rho=1)
print("HDGR(rho=1): " + str(crossval(r)))
gp = GaussianProcessRegressor()
print("GaussianProcessRegressor: " + str(crossval(gp)))
lp = LinearRegression()
print("GaussianProcessRegressor: " + str(crossval(lp)))

