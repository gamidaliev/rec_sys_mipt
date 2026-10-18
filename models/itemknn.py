import numpy as np
from scipy import sparse
from .common import Recommender


class ItemKNN(Recommender):
    def __init__(self, neighbors=50, shrink=20):
        self.neighbors = neighbors
        self.shrink = shrink

    def fit(self, x):
        self.x = x.astype(float).tocsr()
        co = (self.x.T @ self.x).tocsc()
        norms = np.sqrt(self.x.power(2).sum(axis=0)).A1
        rows, columns, values = [], [], []
        for target in range(x.shape[1]):
            a, b = co.indptr[target:target + 2]
            source, count = co.indices[a:b], co.data[a:b]
            similarity = count / np.maximum(norms[source] * norms[target], 1e-12)
            similarity *= count / (count + self.shrink)
            similarity[source == target] = 0
            order = np.lexsort((source, -similarity))[:self.neighbors]
            rows.extend(source[order])
            columns.extend([target] * len(order))
            values.extend(similarity[order])
        self.similarity = sparse.csr_matrix((values, (rows, columns)), shape=co.shape)
        return self

    def score(self, users):
        return (self.x[users] @ self.similarity).toarray()
