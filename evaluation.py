import numpy as np
import pandas as pd
from models.common import topk


def ground_truth(frame):
    return {int(u): set(group.item.astype(int))
            for u, group in frame[frame.rating >= 4].groupby('user', sort=True)}


def ranking_metrics(recommendations, relevant, k=10):
    hits = np.array([item in relevant for item in recommendations[:k]], dtype=float)
    recall = hits.sum() / len(relevant) if relevant else 0.
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = discounts[:min(k, len(relevant))].sum()
    ndcg = (hits * discounts).sum() / ideal if ideal else 0.
    return recall, ndcg


def evaluate(model, seen, truth, batch_size=128, k=10):
    users = np.array(sorted(truth), dtype=int)
    records, recommendations = [], []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        recs = model.recommend(batch, seen, k)
        for u, rec in zip(batch, recs):
            recall, ndcg = ranking_metrics(rec, truth[int(u)], k)
            records.append((int(u), recall, ndcg))
            recommendations.extend(rec.tolist())
    per_user = pd.DataFrame(records, columns=['user', 'Recall@10', 'NDCG@10'])
    metrics = per_user[['Recall@10', 'NDCG@10']].mean().to_dict()
    metrics['Coverage@10'] = len(set(recommendations)) / seen.shape[1]
    metrics['eval_users'] = len(users)
    return metrics, per_user


def paired_interval(left, right, metric='NDCG@10', seed=42, repeats=2000):
    joined = left.set_index('user')[[metric]].join(right.set_index('user')[[metric]], lsuffix='_l', rsuffix='_r', how='inner')
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError('Paired bootstrap requires identical users')
    delta = (joined[f'{metric}_l'] - joined[f'{metric}_r']).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    return {'difference': float(delta.mean()), 'lower_95': float(np.quantile(means, .025)),
            'upper_95': float(np.quantile(means, .975)), 'users': len(delta)}
