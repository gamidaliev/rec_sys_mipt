import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .common import Recommender


class EASE(Recommender):
    def __init__(self, reg=100.):
        self.reg = reg

    def fit(self, x):
        self.x = x.astype(float).tocsr()
        gram = (self.x.T @ self.x).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        inverse = cho_solve(cho_factor(gram), np.eye(gram.shape[0]))
        self.coefficients = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.coefficients, 0)
        return self

    def score(self, users):
        return self.x[users] @ self.coefficients
