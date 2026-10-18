import numpy as np
from scipy.special import expit
from .common import Recommender


class FM(Recommender):
    def __init__(self, factors=16, lr=.04, reg=.0001, epochs=12, negatives=3,
                 batch_size=2048, seed=42, use_metadata=True):
        self.factors, self.lr, self.reg = factors, lr, reg
        self.epochs, self.negatives, self.batch_size = epochs, negatives, batch_size
        self.seed, self.use_metadata = seed, use_metadata

    def _features(self, users, items):
        return np.column_stack([users, self.item_features[items]])

    def _predict(self, features):
        embeddings = self.v[features]
        summed = embeddings.sum(axis=1)
        interaction = .5 * (summed ** 2 - (embeddings ** 2).sum(axis=1)).sum(axis=1)
        return self.bias + self.w[features].sum(axis=1) + interaction

    def fit(self, frame, seen, item_features):
        self.n_users, self.n_items = seen.shape
        self.item_features = item_features.copy()
        if not self.use_metadata:
            self.item_features = self.item_features[:, :1]
        n_features = int(max(self.n_users - 1, self.item_features.max())) + 1
        rng = np.random.default_rng(self.seed)
        self.v = rng.normal(0, .02, (n_features, self.factors))
        rng = np.random.default_rng(self.seed + 1)
        self.w = np.zeros(n_features)
        self.bias = 0.
        acc_v = np.full_like(self.v, 1e-6)
        acc_w = np.full_like(self.w, 1e-6)
        acc_b = 1e-6
        users = frame.user.to_numpy(dtype=int)
        items = frame.item.to_numpy(dtype=int)
        labels = (frame.rating.to_numpy() >= 4).astype(float)
        histories = [set(seen.indices[seen.indptr[u]:seen.indptr[u + 1]]) for u in range(self.n_users)]
        self.losses = []
        for _ in range(self.epochs):
            neg_users = np.repeat(users[labels == 1], self.negatives)
            neg_items = rng.integers(self.n_items, size=len(neg_users))
            for j, u in enumerate(neg_users):
                while neg_items[j] in histories[u]:
                    neg_items[j] = rng.integers(self.n_items)
            u_all = np.concatenate([users, neg_users])
            i_all = np.concatenate([items, neg_items])
            y_all = np.concatenate([labels, np.zeros(len(neg_users))])
            permutation = rng.permutation(len(y_all))
            total = 0.
            for begin in range(0, len(permutation), self.batch_size):
                ix = permutation[begin:begin + self.batch_size]
                f = self._features(u_all[ix], i_all[ix])
                logits = self._predict(f)
                y = y_all[ix]
                total += np.sum(np.logaddexp(0, logits) - y * logits)
                delta = (expit(logits) - y) / len(ix)
                v = self.v[f]
                grad_v = delta[:, None, None] * (v.sum(axis=1)[:, None, :] - v)
                grad_v += self.reg * v / len(ix)
                grad_w = delta[:, None] + self.reg * self.w[f] / len(ix)
                unique, inverse = np.unique(f.ravel(), return_inverse=True)
                gv = np.zeros((len(unique), self.factors))
                gw = np.zeros(len(unique))
                np.add.at(gv, inverse, grad_v.reshape(-1, self.factors))
                np.add.at(gw, inverse, grad_w.ravel())
                acc_v[unique] += gv ** 2
                acc_w[unique] += gw ** 2
                self.v[unique] -= self.lr * gv / np.sqrt(acc_v[unique])
                self.w[unique] -= self.lr * gw / np.sqrt(acc_w[unique])
                gb = delta.sum()
                acc_b += gb ** 2
                self.bias -= self.lr * gb / np.sqrt(acc_b)
            self.losses.append(total / len(y_all))
        return self

    def score(self, users):
        # Split the FM interaction into item-only and user-item terms.
        v_item = self.v[self.item_features]
        summed = v_item.sum(axis=1)
        item_bias = self.w[self.item_features].sum(axis=1)
        item_bias += .5 * (summed ** 2 - (v_item ** 2).sum(axis=1)).sum(axis=1)
        return (self.bias + self.w[users, None] + item_bias[None, :]
                + self.v[users] @ summed.T)
