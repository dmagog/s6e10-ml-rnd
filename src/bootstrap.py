"""Стратифицированный bootstrap ROC-AUC для фиксированных OOF-прогнозов."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def bootstrap_auc(target, score, *, repetitions=1000, seed=20261108, confidence=0.95):
    """Пересэмплировать строки внутри каждого класса, не переобучая модели."""
    target = np.asarray(target)
    score = np.asarray(score, dtype=np.float64)
    if target.ndim != 1 or score.ndim != 1 or len(target) != len(score):
        raise ValueError('Метки и скоры должны быть одномерными массивами одинаковой длины')
    if not np.isfinite(score).all() or set(np.unique(target)) != {0, 1}:
        raise ValueError('Нужны конечные скоры и оба класса 0 и 1')
    if repetitions < 2 or not 0 < confidence < 1:
        raise ValueError('Число повторов должно быть больше одного, уровень доверия — между 0 и 1')

    # Одинаковым скорам соответствует одна группа: такие пары дают половину балла.
    order = np.argsort(score, kind='stable')
    sorted_scores = score[order]
    group = np.r_[0, np.cumsum(sorted_scores[1:] != sorted_scores[:-1])]
    sorted_target = target[order]
    groups_negative = group[sorted_target == 0]
    groups_positive = group[sorted_target == 1]
    n_negative, n_positive = len(groups_negative), len(groups_positive)
    n_groups = int(group[-1]) + 1
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        neg = np.bincount(groups_negative[rng.integers(n_negative, size=n_negative)], minlength=n_groups)
        pos = np.bincount(groups_positive[rng.integers(n_positive, size=n_positive)], minlength=n_groups)
        estimates[i] = np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (n_negative * n_positive)

    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(estimates, [alpha, 1 - alpha], method='linear')
    return {'point_auc': float(roc_auc_score(target, score)), 'lower': float(lower), 'upper': float(upper),
            'confidence': confidence, 'repetitions': repetitions, 'seed': seed,
            'method': 'stratified_percentile_fixed_oof', 'n_positive': n_positive, 'n_negative': n_negative}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('oof', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repetitions', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261108)
    args = parser.parse_args()
    frame = pd.read_csv(args.oof)
    if frame['id'].duplicated().any() or frame[['target', 'score']].isna().any().any():
        raise ValueError('Некорректный файл OOF-прогнозов')
    result = bootstrap_auc(frame['target'], frame['score'], repetitions=args.repetitions, seed=args.seed)
    sha = hashlib.sha256()
    with args.oof.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    result['oof_sha256'] = sha.hexdigest()
    text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
