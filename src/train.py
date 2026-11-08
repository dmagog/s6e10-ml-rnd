"""Воспроизводимая оценка baseline на исходном train S6E10."""
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, roc_auc_score,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
NUMS = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
        'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
        'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
        'Baggage handling', 'Checkin service', 'Cleanliness', 'Departure Delay in Minutes',
        'Arrival Delay in Minutes']

def build_pipeline(cfg):
    """Построить baseline с обучаемой внутри фолда предобработкой."""
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    prep = ColumnTransformer([('numeric', numeric, NUMS), ('categorical', categorical, CATS)])
    return Pipeline([('preprocess', prep),
                     ('model', LogisticRegression(**cfg['model'], random_state=cfg['seed']))])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baseline.yaml')
    parser.add_argument('--data', help='Путь к официальному train.csv')
    parser.add_argument('--output', help='Каталог результатов')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    path = Path(args.data or cfg['data_path'])
    out = Path(args.output or cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(chunk)
    df = pd.read_csv(path)
    required = set(NUMS + CATS + ['id', 'satisfaction'])
    if set(df.columns) != required:
        raise ValueError(f'Неожиданные столбцы: отсутствуют={required-set(df.columns)}, лишние={set(df.columns)-required}')
    if df['id'].duplicated().any() or df['satisfaction'].isna().any():
        raise ValueError('Найдены повторяющиеся id или пропуски целевой переменной')
    target = df['satisfaction'].astype(str).str.lower().map({'true': 1, 'false': 0})
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError('Ожидается бинарная цель True/False')
    X = df[NUMS + CATS].copy()
    y = target.to_numpy(dtype=int)
    n = len(df)
    model = build_pipeline(cfg)
    cv = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=cfg['seed'])
    oof = np.full(n, np.nan)
    folds = np.full(n, -1)
    scores = []
    for i, (tr, va) in enumerate(cv.split(X, y), 1):
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(X.iloc[tr], y[tr])
        prob = model.predict_proba(X.iloc[va])[:, 1]
        if np.isfinite(oof[va]).any():
            raise AssertionError('Повторная оценка строки')
        oof[va], folds[va] = prob, i
        row = {'fold': i, 'n_train': len(tr), 'n_validation': len(va),
               'roc_auc': roc_auc_score(y[va], prob), 'average_precision': average_precision_score(y[va], prob),
               'iterations': int(model.named_steps['model'].n_iter_[0]), 'seconds': round(time.perf_counter()-t, 2)}
        scores.append(row)
        print(json.dumps(row), flush=True)
    assert np.isfinite(oof).all() and (folds > 0).all()
    pd.DataFrame({'id': df.id, 'fold': folds, 'target': y, 'score': oof}).to_csv(out / 'oof.csv', index=False)
    groups = df.groupby(NUMS + CATS, dropna=False)['satisfaction'].agg(['size', 'nunique'])
    duplicated = groups[groups['size'] > 1]
    conflicting = duplicated[duplicated['nunique'] > 1]
    cm = confusion_matrix(y, oof >= cfg['threshold']).tolist()
    errors = df.loc[((y == 0) & (oof >= cfg['threshold'])) | ((y == 1) & (oof < cfg['threshold']))].copy()
    errors['score'] = oof[errors.index]
    errors['error_type'] = np.where(errors['satisfaction'].astype(str).str.lower() == 'true', 'FN', 'FP')
    # Медианные ошибки двух типов и наиболее уверенная ложноотрицательная ошибка.
    fp = errors[errors.error_type == 'FP'].sort_values(['score', 'id'])
    fn = errors[errors.error_type == 'FN'].sort_values(['score', 'id'])
    examples = pd.concat([fp.iloc[[len(fp)//2]], fn.iloc[[len(fn)//2]], fn.iloc[[0]]])
    examples[['id', 'error_type', 'satisfaction', 'score', 'Class', 'Type of Travel', 'Inflight wifi service', 'Seat comfort', 'Inflight entertainment']].to_csv(out / 'error_examples.csv', index=False)
    slices = []
    for col in ['Class', 'Type of Travel']:
        for val, idx in df.groupby(col, dropna=False).groups.items():
            ix = np.asarray(list(idx))
            slices.append({'feature': col, 'value': str(val), 'n': len(ix), 'positive_rate': float(y[ix].mean()),
                           'roc_auc': float(roc_auc_score(y[ix], oof[ix])),
                           'threshold': cfg['threshold'],
                           'error_rate': float(((oof[ix] >= cfg['threshold']) != y[ix]).mean())})
    pd.DataFrame(slices).to_csv(out / 'slices.csv', index=False)
    metrics = {'input_sha256': sha.hexdigest(), 'n': n, 'class_counts': {'false': int((y == 0).sum()), 'true': int((y == 1).sum())},
      'missing': df.isna().sum()[lambda s: s > 0].to_dict(), 'full_duplicate_rows_excluding_id': int(df.drop(columns='id').duplicated().sum()),
      'same_features_groups': int(len(duplicated)), 'same_features_rows': int(duplicated['size'].sum()),
      'conflicting_target_groups': int(len(conflicting)), 'conflicting_target_rows': int(conflicting['size'].sum()),
      'folds': scores, 'mean_fold_auc': float(np.mean([x['roc_auc'] for x in scores])),
      'std_fold_auc_population': float(np.std([x['roc_auc'] for x in scores])),
      'oof_auc': float(roc_auc_score(y, oof)), 'oof_average_precision': float(average_precision_score(y, oof)),
      'confusion_matrix_rows_true_columns_pred': cm, 'threshold': cfg['threshold'],
      'seconds_total': round(time.perf_counter()-start, 2), 'python': platform.python_version(), 'sklearn': sklearn.__version__,
      'input_filename': path.name, 'output_directory_name': out.name,
      'experiment_config': {k: v for k, v in cfg.items() if k not in ('data_path', 'output_dir')}}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+'\n')
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y, oof, ax=ax[0], name='OOF'); ax[0].plot([0, 1], [0, 1], '--', color='grey'); ax[0].set(title='ROC-кривая', xlabel='Доля ложноположительных', ylabel='Доля истинно положительных')
    PrecisionRecallDisplay.from_predictions(y, oof, ax=ax[1], name='OOF'); ax[1].axhline(y.mean(), linestyle='--', color='grey', label='Доля положительных'); ax[1].legend(); ax[1].set(title='Precision–recall', xlabel='Полнота', ylabel='Точность')
    fig.tight_layout(); fig.savefig(out / 'curves.png', dpi=150); plt.close(fig)
    print(json.dumps({'mean_fold_auc': metrics['mean_fold_auc'], 'oof_auc': metrics['oof_auc'], 'seconds': metrics['seconds_total']}), flush=True)

if __name__ == '__main__':
    main()
