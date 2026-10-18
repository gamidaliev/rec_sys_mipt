import numpy as np


def topk(scores, seen, k=10):
    scores = np.array(scores, dtype=float, copy=True)
    rows, cols = seen.nonzero()
    scores[rows, cols] = -np.inf
    # Stable sort makes ties reproducible in ascending item-ID order.
    order = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    if not np.isfinite(np.take_along_axis(scores, order, axis=1)).all():
        raise ValueError('Fewer than k unseen candidates')
    return order


class Recommender:
    def recommend(self, users, seen, k=10):
        return topk(self.score(np.asarray(users)), seen[users], k)
