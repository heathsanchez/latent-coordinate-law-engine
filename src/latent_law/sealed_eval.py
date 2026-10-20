"""Sealed held-out evaluation of simple coordinate candidates (experimental V1).

All feature construction is row-local and label-free; outer test labels are not
read during feature selection. Do not treat this as a replacement for A++.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def candidates(frame, columns):
    x = frame[list(columns)].astype(float).copy()
    for a, b in combinations(columns, 2):
        x[f'{a}__plus__{b}'] = x[a] + x[b]
        x[f'{a}__times__{b}'] = x[a] * x[b]
        x[f'{a}__minus__{b}'] = x[a] - x[b]
        x[f'{a}__over__{b}'] = (x[a] / x[b].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    return x

def sealed_evaluate(df, target, raw_columns, seed=23, max_pairs=500):
    """Select single/pair coordinates on an inner split; score untouched outer test."""
    if target in raw_columns or any(col.startswith('coord_') for col in raw_columns):
        raise ValueError('raw columns must exclude target and precomputed coordinates')
    raw = df[list(raw_columns)].copy()
    y = df[target].copy()
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=.25, random_state=seed, stratify=y)
    tr, va = train_test_split(idx_train, test_size=.30, random_state=seed + 1, stratify=y.iloc[idx_train])
    x = candidates(raw, raw_columns)
    names = list(x.columns)
    combos = [(c,) for c in names] + list(combinations(names, 2))[:max_pairs]
    def fit(cols, rows):
        model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=seed)
        model.fit(x.iloc[rows][list(cols)], y.iloc[rows])
        return model
    scores = []
    for cols in combos:
        model = fit(cols, tr)
        scores.append((accuracy_score(y.iloc[va], model.predict(x.iloc[va][list(cols)])), cols))
    best = max(scores, key=lambda z: (z[0], -len(z[1]), z[1]))
    winner = fit(best[1], idx_train)
    baseline = fit(tuple(raw_columns), idx_train)
    return {
        'selection_inner_accuracy': float(best[0]),
        'selected': list(best[1]),
        'test_accuracy': float(accuracy_score(y.iloc[idx_test], winner.predict(x.iloc[idx_test][list(best[1])] ))),
        'raw_test_accuracy': float(accuracy_score(y.iloc[idx_test], baseline.predict(x.iloc[idx_test][list(raw_columns)]))),
        'train_n': len(idx_train), 'test_n': len(idx_test)
    }
