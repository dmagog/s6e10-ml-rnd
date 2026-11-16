"""Пять фиксированных вариантов и две абляции на фолдах ДЗ 1."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.train import CATS, NUMS, build_pipeline


class AddInteractions(BaseEstimator, TransformerMixin):
    """Построчные произведения оценок без статистик по данным и меток."""
    def __init__(self, definitions):
        self.definitions = definitions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        for item in self.definitions:
            result[item['name']] = result[item['left']] * result[item['right']]
        return result


def make_model(base_cfg, cfg, features):
    if features is None:
        return build_pipeline(base_cfg)
    definitions = [cfg['features'][key] for key in features]
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    prep = ColumnTransformer([('numeric', numeric, NUMS + [x['name'] for x in definitions]),
                              ('categorical', categorical, CATS)])
    return Pipeline([('features', AddInteractions(definitions)), ('preprocess', prep),
                     ('model', lgb.LGBMClassifier(**cfg['lightgbm']))])


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hw2.yaml')
    parser.add_argument('--data')
    parser.add_argument('--output')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path('configs/baseline.yaml').read_text())
    path = Path(args.data or cfg['data_path'])
    out = Path(args.output or cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    reference = json.loads(Path('reports/metrics.json').read_text())
    baseline_oof_sha = json.loads(Path('reports/bootstrap.json').read_text())['oof_sha256']
    if file_sha(path) != reference['input_sha256']:
        raise ValueError('Хеш train.csv отличается от файла ДЗ 1')
    if cfg['n_splits'] != base['n_splits'] or cfg['split_seed'] != base['seed']:
        raise ValueError('Разбиение должно совпадать с ДЗ 1')
    df = pd.read_csv(path)
    if set(df.columns) != set(NUMS + CATS + ['id', 'satisfaction']) or df.id.duplicated().any():
        raise ValueError('Неожиданная схема или повторяющиеся id')
    y = df.satisfaction.astype(str).str.lower().map({'true': 1, 'false': 0}).to_numpy()
    if pd.isna(y).any() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Ожидается бинарная цель True/False')
    X = df[NUMS + CATS]
    splits = list(StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True,
                                  random_state=cfg['split_seed']).split(X, y))
    fold_id = np.zeros(len(df), dtype=np.int8)
    partition = []
    for i, (tr, va) in enumerate(splits, 1):
        fold_id[va] = i
        partition.append({'fold': i, 'train_ids_sha256': ids_sha(df.id.iloc[tr]),
                          'val_ids_sha256': ids_sha(df.id.iloc[va]),
                          'n_train': len(tr), 'n_val': len(va)})
    assert (fold_id > 0).all()
    specs = [('E0', None), ('E1', []), ('E2', ['A']), ('E3', ['B']), ('E4', ['A', 'B'])]
    folds, summaries = [], []
    log_path = out / 'runs.jsonl'
    log_path.write_text('')
    for experiment, features in specs:
        predictions = np.full(len(df), np.nan)
        for i, (tr, va) in enumerate(splits, 1):
            model = make_model(base, cfg, features)
            start = time.perf_counter()
            model.fit(X.iloc[tr], y[tr])
            fit_seconds = time.perf_counter() - start
            scores = model.predict_proba(X.iloc[va])[:, 1]
            if not np.isfinite(scores).all() or np.isfinite(predictions[va]).any():
                raise ValueError('Некорректные или повторные OOF-прогнозы')
            predictions[va] = scores
            record = {'experiment': experiment, 'model': 'logistic' if features is None else 'lightgbm',
                      'features': [] if features is None else features, 'fold': i,
                      'roc_auc': float(roc_auc_score(y[va], scores)),
                      'average_precision': float(average_precision_score(y[va], scores)),
                      'fit_seconds': round(fit_seconds, 3), 'split_seed': cfg['split_seed'],
                      'train_ids_sha256': partition[i-1]['train_ids_sha256'],
                      'val_ids_sha256': partition[i-1]['val_ids_sha256']}
            folds.append(record)
            with log_path.open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if not np.isfinite(predictions).all():
            raise ValueError('Не все строки получили OOF-прогноз')
        oof_path = out / f'{experiment}_oof.csv'
        pd.DataFrame({'id': df.id, 'fold': fold_id, 'target': y, 'score': predictions}).to_csv(oof_path, index=False)
        if experiment == 'E0':
            if file_sha(oof_path) != baseline_oof_sha:
                print('Предупреждение: хеш OOF baseline отличается от ДЗ 1; сравниваем метрики с допуском 1e-6.', flush=True)
            for i, orig in enumerate(reference['folds'], 1):
                actual = next(x for x in folds if x['experiment'] == 'E0' and x['fold'] == i)
                if not np.isclose(actual['roc_auc'], orig['roc_auc'], rtol=0, atol=1e-6):
                    raise ValueError(f'Метрика baseline в фолде {i} не совпала с ДЗ 1')
        row = [x for x in folds if x['experiment'] == experiment]
        summaries.append({'experiment': experiment, 'model': row[0]['model'],
                          'features': '+'.join(features or []) or 'нет',
                          'mean_fold_auc': float(np.mean([x['roc_auc'] for x in row])),
                          'std_fold_auc': float(np.std([x['roc_auc'] for x in row])),
                          'pooled_oof_auc': float(roc_auc_score(y, predictions)),
                          'mean_fold_ap': float(np.mean([x['average_precision'] for x in row])),
                          'fit_seconds_total': round(sum(x['fit_seconds'] for x in row), 3),
                          'oof_sha256': file_sha(oof_path)})
    fold_df = pd.DataFrame(folds)
    fold_df.to_csv(out / 'folds.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    pairs = [('model', 'E1', 'E0'), ('both_features', 'E4', 'E1'),
             ('ablate_A', 'E4', 'E3'), ('ablate_B', 'E4', 'E2')]
    comparisons = []
    for name, left, right in pairs:
        a = fold_df[fold_df.experiment == left].set_index('fold').sort_index()
        b = fold_df[fold_df.experiment == right].set_index('fold').sort_index()
        if not (a.val_ids_sha256 == b.val_ids_sha256).all():
            raise ValueError('Сравнение использует разные проверочные строки')
        d = a.roc_auc - b.roc_auc
        comparisons.append({'comparison': name, 'left': left, 'right': right,
                            'mean_auc_delta': float(d.mean()), 'std_auc_delta': float(d.std(ddof=0)),
                            'positive_folds': int((d > 0).sum()),
                            **{f'fold_{i}_delta': float(d.loc[i]) for i in d.index}})
    pd.DataFrame(comparisons).to_csv(out / 'comparisons.csv', index=False)
    metadata = {'input_sha256': file_sha(path), 'baseline_oof_sha256': baseline_oof_sha,
                'lightgbm_version': lgb.__version__, 'config': cfg, 'partitions': partition,
                'python': os.sys.version.split()[0]}
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    print('Все эксперименты завершены; результаты:', out, flush=True)


if __name__ == '__main__':
    main()
