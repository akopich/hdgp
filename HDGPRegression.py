from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from scipy.optimize import minimize
from scipy import linalg as la


class HDGRegression(BaseEstimator, RegressorMixin):
    def __init__(self, rho = 1, sigma2=0.001):
        self.rho = rho
        self.sigma2 = sigma2
        self.B = None
        self.BOld = np.zeros((1, 1))
        self.kOld = None

    def kernelMatrix(self, X, B):
        if (np.allclose(B, self.BOld)):
            return self.kOld

        B2 = B ** 2
        nonRegularized = (self.deltas * B2[np.newaxis, np.newaxis, :]).sum(axis=2)
        nonRegularized = np.exp(-nonRegularized / 2)

        self.kOld = nonRegularized + np.identity(self.n) * self.sigma2
        self.BOld = B
        return self.kOld

    def kernelDerivative(self, X, K, B, component):
        deltaSquared = self.deltaSquareds[component]
        return -B[component] * deltaSquared * K

    def likelihoodDerivative(self, X, y, B):
        K = self.kernelMatrix(X, B)
        Kinv = la.inv(K)
        alpha = Kinv @ y
        toBeMultipliedWithKernelDerivative = alpha @ np.transpose(alpha) - Kinv

        return -np.array([(toBeMultipliedWithKernelDerivative *
                         self.kernelDerivative(X, K, B, component)).sum()
                         for component in range(0, X.shape[1])])/2 + 2 * self.rho * B

    def likelihood(self, X, y, B):
        K = self.kernelMatrix(X, B)
        Kinv = la.inv(K)

        res =  (np.transpose(y) @ Kinv @ y)[0, 0] / 2 \
               + np.linalg.slogdet(K)[1] / 2 \
               + self.rho * (B**2).sum()
        return res

    def fit(self, X, y=None):
        self.n, self.d = X.shape
        self.deltaSquareds = [np.array([[(X[i, component] - X[j, component]) ** 2
                                         for i in range(0, self.n)]
                                        for j in range(0, self.n)])
                              for component in range(0, self.d)]

        self.deltas = np.array([[X[i, :] - X[j, :]
                                 for i in range(0, self.n)]
                                for j in range(0, self.n)]) ** 2
        B = np.array([1] * X.shape[1])

        self.result = minimize(lambda B: self.likelihood(X, y, B),
                               B,
                               jac=lambda B: self.likelihoodDerivative(X, y, B),
                               options={'gtol': 1e-3*self.d})

        self.B = self.result.x
        print(self.B)
        self.alpha = np.linalg.inv(self.kernelMatrix(X, self.B)) @ y
        self.X = X
        return self

    def crossKernel(self, X, Xtrain):
        result = np.zeros((X.shape[0], Xtrain.shape[0]))
        for i in range(0, X.shape[0]):
            for j in range(0, Xtrain.shape[0]):
                result[i][j] = (((X[i, :] - Xtrain[j, :])*self.B)**2).sum()
        return np.exp(-result / 2)


    def predict(self, Xtest, y=None):
        kstar = self.crossKernel(self.X, Xtest)
        return kstar.T @ self.alpha


