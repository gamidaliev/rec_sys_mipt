import gzip
import ast
import hashlib
import html
import json
import os
import re
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

SEED = 42
TRAIN_END = 1388102400  # 2013-12-27 UTC
VALID_END = 1397001600  # 2014-04-09 UTC
SOURCE = 'https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html'
BASE = 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'
FILES = {
    'reviews_Beauty_5.json.gz': ('', 'e5925ec99023f1dc9c7d3dff7ef34cfeacd40d1b981b5525e1e4ba9c8abc18fe'),
    'meta_Beauty.json.gz': ('', 'c7977dc0e0ead14ac7df8c1c6de74da00e4bebfad87a0f0c5763d4c5bc9b53a0'),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_data():
    root = Path(os.environ.get('REC_SYS_DATA', Path.home() / '.cache/rec_sys_mipt'))
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, (folder, expected) in FILES.items():
        path = root / name
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.part')
            urllib.request.urlretrieve(BASE + folder + name, partial)
            partial.replace(path)
        hashes[name] = sha256(path)
        if expected and hashes[name] != expected:
            raise ValueError(f'Unexpected checksum: {name}')
    return root, hashes


def load_reviews(root):
    records = []
    for line in gzip.open(root / 'reviews_Beauty_5.json.gz', 'rt', encoding='utf-8'):
        d = json.loads(line)
        records.append((d.get('reviewerID'), d.get('asin'), d.get('overall'), d.get('unixReviewTime')))
    return pd.DataFrame(records, columns=['user_id', 'item_id', 'rating', 'timestamp'])


def prepare(raw):
    missing = raw.isna().sum().to_dict()
    frame = raw.dropna().copy()
    frame = frame[frame.rating.between(1, 5) & (frame.timestamp > 0)]
    frame.timestamp = frame.timestamp.astype('int64')
    # Keep the first review, so a later update cannot alter a training label.
    frame = frame.sort_values(['timestamp', 'user_id', 'item_id', 'rating'], kind='stable')
    duplicates = int(frame.duplicated(['user_id', 'item_id']).sum())
    frame = frame.drop_duplicates(['user_id', 'item_id'], keep='first')
    clean = frame.copy()
    initial_train = frame[frame.timestamp < TRAIN_END]
    popularity = initial_train.groupby('item_id').size().sort_values(ascending=False, kind='stable')
    catalog = set(popularity.head(5000).index)
    initial_train = initial_train[initial_train.item_id.isin(catalog)]
    counts = initial_train.groupby('user_id').size()
    users = sorted(counts[counts >= 3].index)
    train_items = sorted(initial_train[initial_train.user_id.isin(users)].item_id.unique())
    selected = frame[frame.user_id.isin(users) & frame.item_id.isin(train_items)].copy()
    user_map = {value: i for i, value in enumerate(users)}
    item_map = {value: i for i, value in enumerate(train_items)}
    selected['user'] = selected.user_id.map(user_map).astype(int)
    selected['item'] = selected.item_id.map(item_map).astype(int)
    train = selected[selected.timestamp < TRAIN_END].copy()
    valid = selected[(selected.timestamp >= TRAIN_END) & (selected.timestamp < VALID_END)].copy()
    test = selected[selected.timestamp >= VALID_END].copy()
    report = {
        'raw_rows': len(raw), 'raw_users': raw.user_id.nunique(), 'raw_items': raw.item_id.nunique(),
        'missing': missing, 'pair_duplicates': duplicates, 'clean_rows': len(clean),
        'selected_rows': len(selected), 'users': len(users), 'items': len(train_items),
        'excluded_rows': len(clean) - len(selected),
        'excluded_cold_user_rows': int((~clean.user_id.isin(users)).sum()),
        'excluded_item_rows_among_warm_users': int((clean.user_id.isin(users) & ~clean.item_id.isin(train_items)).sum()),
    }
    assert 100_000 <= len(selected) <= 2_000_000
    assert len(users) >= 1000 and len(train_items) >= 1000
    assert train.timestamp.max() < valid.timestamp.min() <= valid.timestamp.max() < test.timestamp.min()
    return clean, selected, train, valid, test, users, train_items, report


def matrix(frame, shape):
    x = sparse.csr_matrix((np.ones(len(frame)), (frame.user, frame.item)), shape=shape)
    x.sort_indices()
    return x


def _text(value):
    return re.sub(r'\s+', ' ', html.unescape(str(value or ''))).strip()


def load_metadata(root, items):
    wanted = set(items)
    records, count, duplicates = {}, 0, 0
    for line in gzip.open(root / 'meta_Beauty.json.gz', 'rt', encoding='utf-8'):
        d = ast.literal_eval(line)
        count += 1
        asin = d.get('asin')
        if asin in wanted:
            if asin in records:
                duplicates += 1
                continue
            records[asin] = {'item_id': asin, 'brand': _text(d.get('brand')),
                             'title': _text(d.get('title')), 'category': d.get('categories', [])}
    meta = pd.DataFrame([records.get(asin, {'item_id': asin, 'brand': '', 'title': '', 'category': []}) for asin in items])
    meta['has_metadata'] = meta.item_id.isin(records)
    return meta, {'source_metadata_rows': count, 'duplicate_selected_metadata': duplicates,
                  'metadata_coverage': float(meta.has_metadata.mean())}


def encode_metadata(meta, n_users):
    # Stable catalogue descriptors; no prices, ranks, review text or co-buy links.
    fields = [meta.brand.replace('', '<missing>')]
    for level in [1, 2]:
        fields.append(meta.category.map(
            lambda paths: _text(paths[0][level]) if paths and len(paths[0]) > level else '<missing>'))
    features = [n_users + np.arange(len(meta))]
    offset = n_users + len(meta)
    sizes = {}
    for name, field in zip(['brand', 'category_1', 'category_2'], fields):
        vocab = {value: j for j, value in enumerate(sorted(field.unique()))}
        features.append(offset + field.map(vocab).to_numpy())
        offset += len(vocab)
        sizes[name] = len(vocab)
    return np.column_stack(features).astype(int), sizes
