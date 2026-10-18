import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from data_utils import matrix, TRAIN_END, VALID_END
from evaluation import ranking_metrics
from models import ItemKNN, IALS, EASE, FM, RatingRecencyALS
from models.common import topk


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.x = sparse.csr_matrix([[1., 1., 0., 0.], [1., 0., 1., 0.], [0., 1., 1., 1.]])

    def test_metrics_and_seen_filter(self):
        recall, ndcg = ranking_metrics([2, 4, 1, 0, 3, 5, 6, 7, 8, 9], {2, 1})
        self.assertEqual(recall, 1.)
        self.assertAlmostEqual(ndcg, (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3)))
        seen = sparse.csr_matrix([[1, 0, 0, 0]])
        self.assertEqual(topk([[100., 1., 1., 0.]], seen, 2).tolist(), [[1, 2]])

    def test_ease_matches_constrained_optimum(self):
        model = EASE(reg=2).fit(self.x)
        gram = (self.x.T @ self.x).toarray()
        for j in range(4):
            ids = np.delete(np.arange(4), j)
            expected = np.linalg.solve(gram[np.ix_(ids, ids)] + 2 * np.eye(3), gram[ids, j])
            np.testing.assert_allclose(model.coefficients[ids, j], expected, atol=1e-10)
            self.assertEqual(model.coefficients[j, j], 0)

    def test_als_step_satisfies_normal_equations(self):
        model = IALS(factors=2, reg=2, alpha=3)
        fixed = np.random.default_rng(42).normal(size=(4, 2))
        p = self.x.copy()
        p.data = np.linspace(-1, 1, p.nnz)
        learned = model._step(p, fixed)
        for row in range(3):
            c = np.ones(4)
            c[p[row].indices] += 3
            a = fixed.T @ (c[:, None] * fixed) + 2 * np.eye(2)
            b = fixed.T @ (c * p[row].toarray().ravel())
            np.testing.assert_allclose(a @ learned[row], b, atol=1e-10)

    def test_custom_changes_target_with_equal_confidence(self):
        frame = pd.DataFrame({'user': [0, 0, 0], 'item': [0, 1, 2],
                              'rating': [1., 3., 5.], 'timestamp': [0, 86400, 2 * 86400]})
        model = RatingRecencyALS(half_life=1)
        target = model.make_target(frame, (1, 4), 2 * 86400)
        np.testing.assert_allclose(target.data, [-.625, 0., 1.])
        self.assertEqual(target.nnz, 3)
        self.assertEqual(model.alpha, IALS().alpha)

    def test_fm_score_matches_explicit_feature_interactions(self):
        model = FM(factors=3)
        rng = np.random.default_rng(5)
        model.v = rng.normal(size=(12, 3))
        model.w = rng.normal(size=12)
        model.bias = .2
        model.item_features = np.array([[3, 7, 10], [4, 8, 11]])
        actual = model.score(np.array([0, 1]))
        for u in [0, 1]:
            for item in [0, 1]:
                f = [u] + model.item_features[item].tolist()
                expected = model.bias + model.w[f].sum()
                expected += sum(model.v[a] @ model.v[b] for j, a in enumerate(f) for b in f[j + 1:])
                self.assertAlmostEqual(actual[u, item], expected)
                self.assertAlmostEqual(model._predict(np.array([f]))[0], expected)

    def test_fm_gradient_finite_difference(self):
        rng = np.random.default_rng(42)
        model = FM(factors=2)
        model.v = rng.normal(size=(6, 2))
        model.w = np.zeros(6)
        model.bias = 0.
        f = np.array([[0, 2, 4], [1, 3, 4]])
        labels = np.array([1., 0.])
        logits = model._predict(f)
        delta = (1 / (1 + np.exp(-logits)) - labels) / len(f)
        v = model.v[f]
        gv = delta[:, None, None] * (v.sum(axis=1)[:, None, :] - v)
        analytic = np.zeros_like(model.v)
        np.add.at(analytic, f.ravel(), gv.reshape(-1, 2))
        def loss():
            p = model._predict(f)
            return (np.logaddexp(0, p) - labels * p).mean()
        for i in [0, 2, 4]:
            for j in range(2):
                original = model.v[i, j]
                model.v[i, j] = original + 1e-6
                upper = loss()
                model.v[i, j] = original - 1e-6
                lower = loss()
                model.v[i, j] = original
                self.assertAlmostEqual(analytic[i, j], (upper - lower) / 2e-6, places=6)

    def test_fm_training_is_finite_and_reproducible(self):
        users, items = self.x.nonzero()
        frame = pd.DataFrame({'user': users, 'item': items, 'rating': np.full(len(users), 5.)})
        features = np.column_stack([3 + np.arange(4), 7 + np.arange(4) % 2])
        a = FM(factors=3, epochs=5, batch_size=4).fit(frame, self.x, features)
        b = FM(factors=3, epochs=5, batch_size=4).fit(frame, self.x, features)
        np.testing.assert_array_equal(a.score(np.arange(3)), b.score(np.arange(3)))
        self.assertTrue(np.isfinite(a.losses).all())
        self.assertLess(a.losses[-1], a.losses[0])

    def test_itemknn_cosine_and_shrinkage(self):
        model = ItemKNN(neighbors=4, shrink=2).fit(self.x)
        dense = self.x.toarray()
        co = dense.T @ dense
        norm = np.linalg.norm(dense, axis=0)
        expected = co / (norm[:, None] * norm[None, :]) * co / (co + 2)
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(model.similarity.toarray(), expected)


if __name__ == '__main__':
    unittest.main()
