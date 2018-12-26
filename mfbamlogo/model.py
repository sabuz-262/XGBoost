import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from .RBF_1D  import RBF, ConstantKernel
import logging

class GaussianProcess:
    def __init__(self, numFidelities, dim):
        self.dim = dim
        # Use an anisotropic kernel
        # (independent length scales for each dimension)
        sqrdExp =  RBF(length_scale=1)
        numHyperParams = self.dim + 1
        self.models, self.isFit, self.xValues, self.yValues = [], [], [], []
        numFidelities = 1
        for _ in range(numFidelities):
            self.models.append(GaussianProcessRegressor(
                                    kernel=sqrdExp,
                                    n_restarts_optimizer=numHyperParams*10))
            self.isFit.append(False)
            self.xValues.append([])
            self.yValues.append([])

    def isValid(self, fidelity):
        return len(self.xValues[fidelity]) >= 2

    def fitModel(self):
        fidelity = 0
        if self.isValid(fidelity) and not self.isFit[fidelity]:
            x = np.atleast_2d(self.xValues[fidelity])
            y = np.array(self.yValues[fidelity]).reshape(-1, 1)
            self.models[fidelity].fit(x, y)
            self.isFit[fidelity] = True

    def addSample(self, x, y, fidelity):
        data = np.append(x, fidelity)
        self.xValues[0].append(data)
        self.yValues[0].append(y)
        self.isFit[0] = False

    def getPrediction(self, x, fidelity):
        self.fitModel()
        data = np.append(x,fidelity)
        mean, std = self.models[0].predict(data.reshape(1,-1),return_std=True)
        return np.array(mean)[:, 0], std

    def plotModel(self, ax, fn, ci):
        assert self.dim == 1
        if not self.isValid(-1):
            return
        import matplotlib.pyplot as plt
        xs = np.linspace(0., 1., 500)
        means, vs = self.getPrediction(xs, -1)
        cs = 1.96 * np.sqrt(vs) # 95% confidence
        lcb, ucb = np.array([ci(x) for x in xs]).T
        ax.set_title('Gaussian Process')
        # ax.plot(xs, means, label='Gaussian Process')
        ax.fill_between(xs, means - cs, means + cs, alpha=.5, color='gray')
        ax.plot(xs, [fn(x) for x in xs], label='f(x)', color='#D95F02')
        ax.plot(xs, lcb, '--', label='LCB', color='#7570B3')
        ax.plot(xs, ucb, '--', label='UCB', color='#1B9E77')
        ax.scatter(self.xValues[-1], self.yValues[-1], label='Samples',
                                                       color='blue')
        ax.legend()
        ax.set_xlim([0., 1.])
