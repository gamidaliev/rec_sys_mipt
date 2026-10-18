import numpy as np
from scipy import sparse
from .common import Recommender


class IALS(Recommender):
    def __init__(self, factors=32, reg=10., alpha=20., iterations=10, seed=42):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def _step(self, target, fixed):
        gram = fixed.T @ fixed + self.reg * np.eye(self.factors)
        result = np.empty((target.shape[0], self.factors))
        for row in range(target.shape[0]):
            start, end = target.indptr[row:row + 2]
            ids, values = target.indices[start:end], target.data[start:end]
            y = fixed[ids]
            a = gram + self.alpha * (y.T @ y)
            b = (1 + self.alpha) * (y.T @ values)
            result[row] = np.linalg.solve(a, b)
        return result

    def fit(self, x, target=None):
        p = x.astype(float).tocsr() if target is None else target.astype(float).tocsr()
        # Explicit zeros remain observed entries, with the same confidence as other reviews.
        if p.shape != x.shape or not np.array_equal(p.indptr, x.indptr) or not np.array_equal(p.indices, x.indices):
            raise ValueError('Target must have exactly the observed CSR structure')
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, .05, (x.shape[1], self.factors))
        pt = p.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors = self._step(p, self.item_factors)
            self.item_factors = self._step(pt, self.user_factors)
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
