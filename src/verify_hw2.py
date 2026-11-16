"""Независимо пересчитать метрики и проверить фолды сохранённых прогнозов."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def id_digest(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/train.csv')
    parser.add_argument('--output', default='outputs/hw2')
    args = parser.parse_args()
    out = Path(args.output)
    manifest = json.loads((out / 'manifest.json').read_text())
    if digest(Path(args.data)) != manifest['input_sha256']:
        raise ValueError('Хеш входных данных не совпадает с манифестом')
    df = pd.read_csv(args.data, usecols=['id', 'satisfaction'])
    y = df.satisfaction.astype(str).str.lower().map({'true': 1, 'false': 0}).to_numpy()
    cfg = yaml.safe_load(Path('configs/hw2.yaml').read_text())
    if cfg != manifest['config']:
        raise ValueError('Конфигурация не совпадает с манифестом')
    splits = list(StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True,
                                  random_state=cfg['split_seed']).split(df.id, y))
    folded = np.zeros(len(df), dtype=np.int8)
    for i, (tr, va) in enumerate(splits, 1):
        folded[va] = i
        part = manifest['partitions'][i-1]
        if id_digest(df.id.iloc[tr]) != part['train_ids_sha256'] or id_digest(df.id.iloc[va]) != part['val_ids_sha256']:
            raise ValueError(f'Состав фолда {i} не совпал')
    folds = pd.read_csv(out / 'folds.csv')
    summary = pd.read_csv(out / 'summary.csv').set_index('experiment')
    for name in ['E0', 'E1', 'E2', 'E3', 'E4']:
        path = out / f'{name}_oof.csv'
        pred = pd.read_csv(path)
        if digest(path) != summary.loc[name, 'oof_sha256']:
            raise ValueError(f'{name}: хеш OOF не совпал')
        if not (np.array_equal(pred.id, df.id) and np.array_equal(pred.target, y) and np.array_equal(pred.fold, folded)):
            raise ValueError(f'{name}: строки, метки или фолды не совпали')
        if not np.isfinite(pred.score).all() or not pred.score.between(0, 1).all():
            raise ValueError(f'{name}: некорректные скоры')
        for i in range(1, cfg['n_splits'] + 1):
            mask = folded == i
            row = folds[(folds.experiment == name) & (folds.fold == i)]
            if len(row) != 1 or not np.isclose(roc_auc_score(y[mask], pred.score[mask]), row.roc_auc.iloc[0], rtol=0, atol=1e-12):
                raise ValueError(f'{name}: ROC-AUC фолда {i} не совпал')
            if not np.isclose(average_precision_score(y[mask], pred.score[mask]), row.average_precision.iloc[0], rtol=0, atol=1e-12):
                raise ValueError(f'{name}: AP фолда {i} не совпала')
        if not np.isclose(roc_auc_score(y, pred.score), summary.loc[name, 'pooled_oof_auc'], rtol=0, atol=1e-12):
            raise ValueError(f'{name}: OOF ROC-AUC не совпал')
    if digest(out / 'E0_oof.csv') != json.loads(Path('reports/bootstrap.json').read_text())['oof_sha256']:
        print('Хеш baseline отличается от ДЗ 1; возможны численные различия между окружениями.')
    baseline = json.loads(Path('reports/metrics.json').read_text())
    for i, original in enumerate(baseline['folds'], 1):
        actual = folds[(folds.experiment == 'E0') & (folds.fold == i)].roc_auc.iloc[0]
        if not np.isclose(actual, original['roc_auc'], rtol=0, atol=1e-6):
            raise ValueError(f'ROC-AUC baseline в фолде {i} отличается от ДЗ 1')
    compare = pd.read_csv(out / 'comparisons.csv')
    for row in compare.itertuples():
        left = folds[folds.experiment == row.left].set_index('fold').sort_index()
        right = folds[folds.experiment == row.right].set_index('fold').sort_index()
        delta = left.roc_auc - right.roc_auc
        if not np.isclose(delta.mean(), row.mean_auc_delta, rtol=0, atol=1e-12):
            raise ValueError(f'{row.comparison}: средняя разница не совпала')
        for i in delta.index:
            if not np.isclose(delta.loc[i], getattr(row, f'fold_{i}_delta'), rtol=0, atol=1e-12):
                raise ValueError(f'{row.comparison}: разница фолда {i} не совпала')
    print('Проверены данные, фолды, 5 OOF-файлов, ROC-AUC, AP и парные разницы.')


if __name__ == '__main__':
    main()
