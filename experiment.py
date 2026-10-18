import gc
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits
from models import ItemKNN, IALS, EASE, FM, RatingRecencyALS
from data_utils import TRAIN_END, VALID_END, matrix
from evaluation import evaluate, ground_truth, paired_interval

SEARCH = {
    'ItemKNN': [{'neighbors': 30, 'shrink': 10}, {'neighbors': 100, 'shrink': 30}],
    'iALS': [{'factors': 32, 'reg': 10., 'alpha': 20., 'iterations': 12},
             {'factors': 64, 'reg': 20., 'alpha': 20., 'iterations': 12}],
    'EASE': [{'reg': 100.}, {'reg': 500.}],
    'FM': [{'factors': 16, 'lr': .04, 'epochs': 12},
           {'factors': 32, 'lr': .04, 'epochs': 12}],
}
CLASSES = {'ItemKNN': ItemKNN, 'iALS': IALS, 'EASE': EASE, 'FM': FM,
           'CustomALS': RatingRecencyALS, 'FM_ID_only': FM, 'RatingALS': RatingRecencyALS}


def fit_model(name, params, frame, x, features, reference_time):
    model = CLASSES[name](**params)
    if isinstance(model, FM):
        return model.fit(frame, x, features)
    if isinstance(model, RatingRecencyALS):
        return model.fit(x, frame, reference_time)
    return model.fit(x)


def tune(train, valid, shape, features, output='results'):
    output = Path(output)
    output.mkdir(exist_ok=True)
    x = matrix(train, shape)
    truth = ground_truth(valid)
    rows, chosen = [], {}
    spaces = dict(SEARCH)
    with threadpool_limits(limits=2):
        for name in ['ItemKNN', 'iALS', 'EASE', 'FM', 'CustomALS']:
            if name == 'CustomALS':
                spaces[name] = [dict(chosen['iALS'], half_life=h) for h in [90, 365]]
            trials = []
            for params in spaces[name]:
                start = time.perf_counter()
                model = fit_model(name, params, train, x, features, TRAIN_END)
                fit_seconds = time.perf_counter() - start
                metrics, _ = evaluate(model, x, truth)
                row = {'model': name, 'params': json.dumps(params, sort_keys=True),
                       'fit_seconds': fit_seconds, **metrics}
                rows.append(row)
                trials.append(row)
                print(name, params, {k: round(v, 5) for k, v in metrics.items()}, flush=True)
                if isinstance(model, FM):
                    (output / f'fm_loss_{params["factors"]}.json').write_text(json.dumps(model.losses))
                del model
                gc.collect()
            # Stable tie break: first configuration in the declared grid.
            best = max(trials, key=lambda row: row['NDCG@10'])
            chosen[name] = json.loads(best['params'])
            pd.DataFrame(rows).to_csv(output / 'validation.csv', index=False)
            (output / 'selected_params.json').write_text(json.dumps(chosen, indent=2))
    return pd.DataFrame(rows), chosen, spaces


def final_evaluation(train_valid, test, shape, features, chosen, output='results'):
    output = Path(output)
    frozen = json.loads(json.dumps(chosen))
    configs = dict(frozen)
    configs['FM_ID_only'] = dict(configs['FM'], use_metadata=False)
    configs['RatingALS'] = dict(configs['CustomALS'], half_life=float('inf'))
    x = matrix(train_valid, shape)
    truth = ground_truth(test)
    rows, all_users, examples = [], {}, []
    with threadpool_limits(limits=2):
        for name, params in configs.items():
            start = time.perf_counter()
            model = fit_model(name, params, train_valid, x, features, VALID_END)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            metrics, per_user = evaluate(model, x, truth)
            row = {'model': name, 'params': json.dumps(params, sort_keys=True),
                   'fit_seconds': fit_seconds, 'ranking_seconds': time.perf_counter() - start,
                   **metrics}
            rows.append(row)
            all_users[name] = per_user
            per_user.to_csv(output / f'users_{name}.csv', index=False)
            # Chosen by user ID, without inspecting model scores or outcomes.
            example_users = sorted(truth)[:3]
            recs = model.recommend(np.array(example_users), x, 10)
            for u, rec in zip(example_users, recs):
                for rank, item in enumerate(rec, 1):
                    examples.append({'model': name, 'user': u, 'rank': rank, 'item': int(item),
                                     'relevant': int(item in truth[u])})
            print('TEST', name, {k: round(v, 5) for k, v in metrics.items()}, flush=True)
            del model
            gc.collect()
    comparisons = {
        'CustomALS - iALS': paired_interval(all_users['CustomALS'], all_users['iALS']),
        'FM - FM_ID_only': paired_interval(all_users['FM'], all_users['FM_ID_only']),
        'CustomALS - RatingALS': paired_interval(all_users['CustomALS'], all_users['RatingALS']),
    }
    result = pd.DataFrame(rows)
    result.to_csv(output / 'test.csv', index=False)
    pd.DataFrame(examples).to_csv(output / 'recommendations.csv', index=False)
    (output / 'bootstrap.json').write_text(json.dumps(comparisons, indent=2))
    (output / 'environment.json').write_text(json.dumps({
        'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
        'scipy': scipy.__version__, 'threads': 2, 'seed': 42,
    }, indent=2))
    assert chosen == frozen
    return result, comparisons
